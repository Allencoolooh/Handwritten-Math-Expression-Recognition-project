# eval_test.py
from pathlib import Path
from typing import List, Tuple, Dict

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
    用于“公式级准确率”的对比：去掉所有空白字符。
    """
    return "".join(s.split())


def latex_len_tokens(s: str) -> int:
    """
    用空格分词估计 LaTeX 公式 token 长度（你的标注已按 token 空格分开）
    """
    return len(s.strip().split()) if s.strip() else 0


def tokenize_latex_space(s: str) -> List[str]:
    """
    你的标注是“空格分 token”，因此这里直接 split 即可。
    """
    s = s.strip()
    return s.split() if s else []


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    """
    token-level Levenshtein 编辑距离
    - 插入/删除/替换代价均为 1
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # 1D DP 省内存
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # delete
                cur[j - 1] + 1,     # insert
                prev[j - 1] + cost  # substitute
            )
        prev = cur
    return prev[m]


def load_best(model: MathFormulaRecognizer):
    ckpt_dir = getattr(Config, "CKPT_DIR", "checkpoints")
    ckpt_path = Path(ckpt_dir) / "mixedft_ss_best.pt"
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
    在【测试集】上评估，并把结果写入 /results：
      - results/test_summary.txt
      - results/test_samples.txt

    新增指标：
      - Near-miss (准成功) 生成率：
          * token 编辑距离 <= {2,3,5}
          * token 错误率 <= {5%,10%}
        只统计那些 “公式不完全正确但接近正确” 的样本。
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
        augment=False,
    )
    print(f"[TestEval] Loaded test set with {len(test_loader.dataset)} samples.")

    # 3. 构建模型 & 加载权重
    model = MathFormulaRecognizer(vocab).to(device)
    load_best(model)
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0

    # 公式级统计（总体）
    total_formulas = 0
    correct_formulas = 0

    # 按长度分桶统计（公式级准确率）
    bucket_stats = {
        "short": {"correct": 0, "total": 0},  # L <= 10
        "mid":   {"correct": 0, "total": 0},  # 10 < L <= 20
        "long":  {"correct": 0, "total": 0},  # L > 20
    }

    # === Near-miss 统计 ===
    # 编辑距离阈值
    ed_thresholds = [2, 3, 5]
    near_by_ed = {k: 0 for k in ed_thresholds}     # 只统计非完全正确样本中的近似正确数
    # 错误率阈值
    er_thresholds = [0.05, 0.10]
    near_by_er = {k: 0 for k in er_thresholds}

    total_non_exact = 0  # 非完全正确的公式数量（near-miss 分母）

    # 打开样本结果文件
    f_samples = samples_path.open("w", encoding="utf-8")

    print("[TestEval] Evaluating on TEST set and saving results to /results ...")

    for batch_id, batch in enumerate(test_loader):
        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)
        labels = batch["labels"]

        if batch_id % 10 == 0:
            print(f"[TestEval] On batch {batch_id}/{len(test_loader)}")

        # 训练模式一致的 token loss（teacher forcing）
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

        # 推理解码
        preds = model.recognize(
            images,
            max_len=getattr(Config, "MAX_TGT_LEN", 128),
            device=device,
        )

        # 逐条统计
        for i, (gt, pr) in enumerate(zip(labels, preds)):
            norm_gt = normalize_formula(gt)
            norm_pr = normalize_formula(pr)
            is_exact = (norm_gt == norm_pr)

            total_formulas += 1
            if is_exact:
                correct_formulas += 1

            # 长度分桶（按 GT token 数）
            L_tokens = latex_len_tokens(gt)
            if L_tokens <= 10:
                bucket = "short"
                bucket_label = "L <= 10"
            elif L_tokens <= 20:
                bucket = "mid"
                bucket_label = "10 < L <= 20"
            else:
                bucket = "long"
                bucket_label = "L > 20"

            bucket_stats[bucket]["total"] += 1
            if is_exact:
                bucket_stats[bucket]["correct"] += 1

            # === Near-miss 计算（只对非 exact 的样本）===
            gt_toks = tokenize_latex_space(gt)
            pr_toks = tokenize_latex_space(pr)

            ed = levenshtein_distance(gt_toks, pr_toks)
            denom = max(len(gt_toks), len(pr_toks), 1)
            err_rate = ed / denom  # 用编辑距离近似“token 错误率”

            near_flags_ed = {}
            near_flags_er = {}

            if not is_exact:
                total_non_exact += 1

                for k in ed_thresholds:
                    ok = (ed <= k)
                    near_flags_ed[k] = ok
                    if ok:
                        near_by_ed[k] += 1

                for r in er_thresholds:
                    ok = (err_rate <= r)
                    near_flags_er[r] = ok
                    if ok:
                        near_by_er[r] += 1

            # 样本写入文件（包含 near-miss 信息）
            f_samples.write(f"[Batch {batch_id} Sample {i}]\n")
            f_samples.write(f"Bucket: {bucket_label}\n")
            f_samples.write(f"ExactCorrect: {is_exact}\n")
            f_samples.write(f"EditDistance(tokens): {ed}\n")
            f_samples.write(f"ErrRate≈ED/maxLen: {err_rate:.4f}\n")
            if not is_exact:
                f_samples.write(f"NearByED: " +
                                ", ".join([f"ED<={k}:{near_flags_ed.get(k, False)}" for k in ed_thresholds]) + "\n")
                f_samples.write(f"NearByER: " +
                                ", ".join([f"ER<={int(r*100)}%:{near_flags_er.get(r, False)}" for r in er_thresholds]) + "\n")
            f_samples.write(f"GT  ({L_tokens} tokens): {gt}\n")
            f_samples.write(f"Pred: {pr}\n\n")

    f_samples.close()

    # 汇总整体指标
    avg_loss = total_loss / max(total_tokens, 1)
    avg_token_acc = total_acc / max(total_tokens, 1)
    formula_acc = correct_formulas / total_formulas if total_formulas > 0 else 0.0

    # 计算 near-miss 率：分母是“非完全正确样本数”
    near_rate_ed = {k: (near_by_ed[k] / total_non_exact if total_non_exact > 0 else 0.0) for k in ed_thresholds}
    near_rate_er = {r: (near_by_er[r] / total_non_exact if total_non_exact > 0 else 0.0) for r in er_thresholds}

    # 写入 summary
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("===== Test Set Summary (evaluated with checkpoint) =====\n")
        f.write(f"Avg loss             : {avg_loss:.4f}\n")
        f.write(f"Token accuracy       : {avg_token_acc*100:.2f}%\n")
        f.write(f"Formula accuracy     : {formula_acc*100:.2f}% ({correct_formulas}/{total_formulas})\n")
        f.write(f"Total formulas       : {total_formulas}\n\n")

        # 各长度桶
        for name, info in bucket_stats.items():
            acc_b = (info["correct"] / info["total"]) if info["total"] > 0 else 0.0
            if name == "short":
                label = "L <= 10"
            elif name == "mid":
                label = "10 < L <= 20"
            else:
                label = "L > 20"

            f.write(
                f"Formula accuracy [{label:12s}]: "
                f"{acc_b*100:.2f}% ({info['correct']}/{info['total']})\n"
            )

        f.write("\n")
        f.write("----- Near-miss (almost correct) -----\n")
        f.write(f"Non-exact formulas    : {total_non_exact}\n")

        # Near-miss by ED
        for k in ed_thresholds:
            f.write(f"Near-miss rate (ED <= {k}) : {near_rate_ed[k]*100:.2f}% ({near_by_ed[k]}/{total_non_exact})\n")

        # Near-miss by error rate
        for r in er_thresholds:
            f.write(f"Near-miss rate (ER <= {int(r*100)}%) : {near_rate_er[r]*100:.2f}% ({near_by_er[r]}/{total_non_exact})\n")

    # 终端打印
    print("\n===== Test Set Summary =====")
    print(f"Avg loss         : {avg_loss:.4f}")
    print(f"Token accuracy   : {avg_token_acc*100:.2f}%")
    print(f"Formula accuracy : {formula_acc*100:.2f}% ({correct_formulas}/{total_formulas})")
    print(f"Non-exact        : {total_non_exact}")

    for name, info in bucket_stats.items():
        acc_b = (info["correct"] / info["total"]) if info["total"] > 0 else 0.0
        if name == "short":
            label = "L <= 10"
        elif name == "mid":
            label = "10 < L <= 20"
        else:
            label = "L > 20"

        print(f"Formula accuracy [{label:12s}]: {acc_b*100:.2f}% ({info['correct']}/{info['total']})")

    print("\n----- Near-miss (almost correct) -----")
    print(f"Non-exact formulas: {total_non_exact}")
    for k in ed_thresholds:
        print(f"Near-miss rate (ED <= {k}) : {near_rate_ed[k]*100:.2f}% ({near_by_ed[k]}/{total_non_exact})")
    for r in er_thresholds:
        print(f"Near-miss rate (ER <= {int(r*100)}%) : {near_rate_er[r]*100:.2f}% ({near_by_er[r]}/{total_non_exact})")

    print(f"\n[Saved] Summary  -> {summary_path}")
    print(f"[Saved] Samples  -> {samples_path}")


if __name__ == "__main__":
    evaluate_test_set()
