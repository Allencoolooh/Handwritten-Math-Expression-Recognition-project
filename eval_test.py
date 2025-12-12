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
    """
    计算 token-level accuracy（忽略 <PAD>）
    """
    preds = logits.argmax(dim=-1)      # (B, L)
    mask = targets.ne(pad_id)          # (B, L)
    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def normalize_formula(s: str) -> str:
    """
    用于“公式级准确率”的对比：
    这里简单地去掉所有空白字符，保留 LaTeX 语义相关字符。
    """
    return "".join(s.split())


def latex_len_tokens(s: str) -> int:
    """
    用空格分词估计 LaTeX 公式 token 长度。
    （你的标注已经是按 token 用空格分开的）
    """
    return len(s.strip().split()) if s.strip() else 0


def load_best(model: MathFormulaRecognizer):
    ckpt_dir = getattr(Config, "CKPT_DIR", "checkpoints")
    ckpt_path = Path(ckpt_dir) / "longft_last_epoch005.pt"
    assert ckpt_path.is_file(), f"[TestEval] best.pt not found at: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    print(f"[TestEval] Loaded best model from {ckpt_path}")


@torch.no_grad()
def evaluate_test_set():
    """
    在【测试集】上评估 best.pt，并把结果写入 /results：
      - results/test_summary.txt
      - results/test_samples.txt
    指标包括：
      - token-level accuracy
      - formula-level accuracy（整句完全一致）
      - 按长度分桶的公式级准确率：
          * L <= 10
          * 10 < L <= 20
          * L > 20
    """
    device = get_device()
    print("[TestEval] Using device:", device)

    # 0. 准备结果目录
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)

    summary_path = result_dir / "test_summary.txt"
    samples_path = result_dir / "test_samples.txt"

    # 1. 加载 vocab
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    pad_id = vocab.pad_id
    print("[TestEval] Vocab size:", len(vocab))

    # 2. 构建【测试集】 DataLoader
    test_loader = create_dataloader(
        labels_path=Config.TEST_LABELS,
        vocab=vocab,
        batch_size=getattr(Config, "BATCH_SIZE", 16),
        shuffle=False,
        num_workers=0,
        augment=False,  # 测试集不做增强
    )
    print(f"[TestEval] Loaded test set with {len(test_loader.dataset)} samples.")

    # 3. 构建模型 & 加载 best.pt
    model = MathFormulaRecognizer(vocab).to(device)
    load_best(model)
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0

    # 公式级统计（总体）
    total_formulas = 0
    correct_formulas = 0

    # 按长度分桶统计
    bucket_stats = {
        "short": {"correct": 0, "total": 0},  # L <= 10
        "mid":   {"correct": 0, "total": 0},  # 10 < L <= 20
        "long":  {"correct": 0, "total": 0},  # L > 20
    }

    # 打开样本结果文件
    f_samples = samples_path.open("w", encoding="utf-8")

    print("[TestEval] Evaluating on TEST set and saving results to /results ...")

    for batch_id, batch in enumerate(test_loader):
        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)
        labels = batch["labels"]  # 原始 LaTeX 文本 list[str]

        if batch_id % 10 == 0:
            print(f"[TestEval] On batch {batch_id}/{len(test_loader)}")

        # 前向
        logits = model(images, tgt_input, tgt_lengths)  # (B, L, V)
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

        # 解码
        preds = model.recognize_beam(
            images,
            max_len=getattr(Config, "MAX_TGT_LEN", 128),
            device=device,
            beam_size=5,
        )

        # 写入每条样本 & 统计公式级准确率（含长度分桶）
        for i, (gt, pr) in enumerate(zip(labels, preds)):
            norm_gt = normalize_formula(gt)
            norm_pr = normalize_formula(pr)
            is_correct = (norm_gt == norm_pr)

            # 总体公式统计
            total_formulas += 1
            if is_correct:
                correct_formulas += 1

            # 按长度分桶（用原始 LaTeX 的空格 token 数）
            L_tokens = latex_len_tokens(gt)
            if L_tokens <= 10:
                bucket = "short"
            elif L_tokens <= 20:
                bucket = "mid"
            else:
                bucket = "long"

            bucket_stats[bucket]["total"] += 1
            if is_correct:
                bucket_stats[bucket]["correct"] += 1

            # 样本写入文件
            f_samples.write(f"[Batch {batch_id} Sample {i}]\n")
            f_samples.write(f"Correct: {is_correct}\n")
            f_samples.write(f"GT  ({L_tokens} tokens): {gt}\n")
            f_samples.write(f"Pred: {pr}\n\n")

    f_samples.close()

    # 4. 汇总整体指标
    avg_loss = total_loss / max(total_tokens, 1)
    avg_token_acc = total_acc / max(total_tokens, 1)
    formula_acc = (
        correct_formulas / total_formulas if total_formulas > 0 else 0.0
    )

    # 5. 写入 summary 文件
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("===== Test Set Summary (evaluated with best.pt) =====\n")
        f.write(f"Avg loss             : {avg_loss:.4f}\n")
        f.write(f"Token accuracy       : {avg_token_acc*100:.2f}%\n")
        f.write(f"Formula accuracy     : {formula_acc*100:.2f}%\n")
        f.write(f"Total formulas       : {total_formulas}\n\n")

        # 各长度桶的公式级准确率
        for name, info in bucket_stats.items():
            if info["total"] == 0:
                acc = 0.0
            else:
                acc = info["correct"] / info["total"]

            if name == "short":
                label = "L <= 10"
            elif name == "mid":
                label = "10 < L <= 20"
            else:
                label = "L > 20"

            f.write(
                f"Formula accuracy [{label:8s}]: "
                f"{acc*100:.2f}% ({info['correct']}/{info['total']})\n"
            )

    # 终端打印
    print("\n===== Test Set Summary =====")
    print(f"Avg loss         : {avg_loss:.4f}")
    print(f"Token accuracy   : {avg_token_acc*100:.2f}%")
    print(f"Formula accuracy : {formula_acc*100:.2f}%")
    print(f"Total formulas   : {total_formulas}")

    for name, info in bucket_stats.items():
        if info["total"] == 0:
            acc = 0.0
        else:
            acc = info["correct"] / info["total"]

        if name == "short":
            label = "L <= 10"
        elif name == "mid":
            label = "10 < L <= 20"
        else:
            label = "L > 20"

        print(
            f"Formula accuracy [{label:8s}]: "
            f"{acc*100:.2f}% ({info['correct']}/{info['total']})"
        )

    print(f"\n[Saved] Summary  -> {summary_path}")
    print(f"[Saved] Samples  -> {samples_path}")


if __name__ == "__main__":
    evaluate_test_set()

'''[Train] Epoch 80 done. Loss 1.0596, Acc 93.62%
[Val]   Epoch 80 done. Loss 1.1272, Acc 92.37%
'''
