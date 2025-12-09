# eval_test.py
from pathlib import Path

import torch
import torch.nn.functional as F

from config import Config
from utils.vocab import Vocab
from utils.dataset import create_dataloader
from model.model import MathFormulaRecognizer


def get_device():
    if hasattr(Config, "DEVICE"):
        return torch.device(Config.DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_token_accuracy(logits, targets, pad_id: int) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets.ne(pad_id)
    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def load_best(model):
    ckpt_dir = getattr(Config, "CKPT_DIR", "checkpoints")
    ckpt_path = Path(ckpt_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    print(f"[Test] Loaded best model from {ckpt_path}")


@torch.no_grad()
def evaluate_test():
    """
    在测试集上评估 best.pt，并将结果保存到 /results 文件夹。
    """
    device = get_device()
    print("[Test] Using device:", device)

    # ========== 0. 准备结果目录 ==========
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)

    summary_path = result_dir / "test_summary.txt"
    samples_path = result_dir / "test_samples.txt"

    # ========== 1. 加载 vocab ==========
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    print("[Test] Vocab size:", len(vocab))

    # ========== 2. 加载 test DataLoader ==========
    test_loader = create_dataloader(
        labels_path=Config.TEST_LABELS,
        vocab=vocab,
        batch_size=getattr(Config, "BATCH_SIZE", 16),
        shuffle=False,
        num_workers=0,
        augment=False,
    )

    # ========== 3. 加载模型 ==========
    model = MathFormulaRecognizer(vocab).to(device)
    load_best(model)
    model.eval()

    pad_id = vocab.pad_id

    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0

    # 打开文件，准备写入
    f_samples = samples_path.open("w", encoding="utf-8")

    print("[Test] Running full test set...")
    for batch_id, batch in enumerate(test_loader):

        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)
        labels = batch["labels"]

        logits = model(images, tgt_input, tgt_lengths)
        B, L, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * L, V),
            tgt_output.view(B * L),
            ignore_index=pad_id,
        )

        acc = compute_token_accuracy(logits, tgt_output, pad_id)
        n_tokens = (tgt_output != pad_id).sum().item()

        total_loss += loss.item() * n_tokens
        total_acc += acc * n_tokens
        total_tokens += n_tokens

        # ======= 预测（greedy） =======
        preds = model.recognize(
            images,
            max_len=getattr(Config, "MAX_TGT_LEN", 128),
            device=device,
        )

        # ======= 写入样本预测结果 =======
        for i, (gt, pr) in enumerate(zip(labels, preds)):
            f_samples.write(f"[Batch {batch_id} Sample {i}]\n")
            f_samples.write(f"GT  : {gt}\n")
            f_samples.write(f"Pred: {pr}\n\n")

    f_samples.close()

    # ========== 4. 计算最终指标 ==========
    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_acc / max(total_tokens, 1)

    # ========== 5. 写入 summary 文件 ==========
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("===== Test Summary =====\n")
        f.write(f"Avg loss       : {avg_loss:.4f}\n")
        f.write(f"Token accuracy : {avg_acc*100:.2f}%\n")

    print("\n===== Test summary =====")
    print(f"Avg loss       : {avg_loss:.4f}")
    print(f"Token accuracy : {avg_acc*100:.2f}%")
    print(f"\n[Saved] Summary  -> {summary_path}")
    print(f"[Saved] Samples  -> {samples_path}")


if __name__ == "__main__":
    evaluate_test()
