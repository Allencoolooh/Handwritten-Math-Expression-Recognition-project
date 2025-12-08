import torch
import torch.nn.functional as F
from pathlib import Path

from config import Config
from utils.vocab import Vocab
from utils.dataset import create_dataloader
from model.model import MathFormulaRecognizer


def load_best_model(vocab: Vocab, ckpt_path: Path) -> MathFormulaRecognizer:
    """
    从 best.pt 加载训练好的模型权重。
    兼容我们之前 train.py 里的 checkpoint 格式：
        {
            "epoch": ...,
            "model_state": ...,
            "optimizer_state": ...,
            "best_val_loss": ...
        }
    """
    device = Config.DEVICE if hasattr(Config, "DEVICE") else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Info] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = MathFormulaRecognizer(vocab).to(device)

    # 兼容两种 key：model_state / model
    state_dict = ckpt.get("model_state", ckpt.get("model", None))
    if state_dict is None:
        raise KeyError("Checkpoint 中未找到 'model_state' 或 'model' 键，请检查保存格式。")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Info] Loaded model (epoch={ckpt.get('epoch', '?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', 'N/A')})")
    return model


@torch.no_grad()
def evaluate_model(
    model: MathFormulaRecognizer,
    data_loader,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float]:
    """
    在 data_loader 上评估：
        返回 (avg_loss, token_accuracy)
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in data_loader:
        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        logits = model(images, tgt_input, tgt_lengths)  # (B, L, V)
        B, L, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * L, V),
            tgt_output.view(B * L),
            ignore_index=pad_id,
        )

        # 统计 token-level 准确率
        preds = logits.argmax(dim=-1)      # (B, L)
        mask = tgt_output.ne(pad_id)       # (B, L)
        correct = (preds.eq(tgt_output) & mask).sum().item()
        tokens = mask.sum().item()

        total_loss += loss.item() * B
        total_correct += correct
        total_tokens += tokens

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_correct / max(total_tokens, 1)

    return avg_loss, avg_acc


@torch.no_grad()
def show_examples(
    model: MathFormulaRecognizer,
    data_loader,
    device: torch.device,
    num_batches: int = 1,
    max_len: int | None = None,
):
    """
    打印若干条 GT vs 预测结果，直观看看效果。
    """
    if max_len is None:
        max_len = getattr(Config, "MAX_TGT_LEN", 128)

    print("\n===== Some examples (GT vs Pred) =====")
    count = 0

    for b_idx, batch in enumerate(data_loader):
        if b_idx >= num_batches:
            break

        images = batch["images"].to(device)
        labels = batch["labels"]   # list[str]

        preds = model.recognize(images, max_len=max_len, device=device)  # list[str]

        for i, (gt, pr) in enumerate(zip(labels, preds)):
            print(f"\n[Sample {count}]")
            print("GT  :", gt)
            print("Pred:", pr)
            count += 1


def main():
    # 设备
    device = Config.DEVICE if hasattr(Config, "DEVICE") else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("[Info] Using device:", device)

    # 1. 加载 vocab
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    print("[Info] Vocab size:", len(vocab))

    # 2. 加载 best.pt
    ckpt_dir = getattr(Config, "CKPT_DIR", Path("checkpoints"))
    ckpt_path = Path(ckpt_dir) / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_best_model(vocab, ckpt_path)

    # 3. 构建验证集 / 测试集 DataLoader
    #    这里先用 VAL_LABELS，你有 test.txt 时可以改成 TEST_LABELS
    eval_labels = Config.VAL_LABELS   # 或者 Config.TEST_LABELS (如果你以后有的话)

    eval_loader = create_dataloader(
        labels_path=eval_labels,
        vocab=vocab,
        batch_size=getattr(Config, "BATCH_SIZE", 16),
        shuffle=False,
        num_workers=getattr(Config, "NUM_WORKERS", 0),
        augment=False,
    )

    # 4. 评估整体 loss + token 准确率
    avg_loss, avg_acc = evaluate_model(
        model,
        eval_loader,
        device=device,
        pad_id=vocab.pad_id,
    )
    print("\n===== Eval result =====")
    print(f"Avg loss: {avg_loss:.4f}")
    print(f"Token accuracy: {avg_acc*100:.2f}%")

    # 5. 打印若干条样例
    show_examples(
        model,
        eval_loader,
        device=device,
        num_batches=1,                    # 看前 1 个 batch 的若干条
        max_len=getattr(Config, "MAX_TGT_LEN", 128),
    )


if __name__ == "__main__":
    main()
