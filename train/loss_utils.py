# train/loss_utils.py
from __future__ import annotations

from typing import Dict, Iterable, Set
import torch
from utils.vocab import Vocab


def _ids_for_tokens(vocab: Vocab, toks: Iterable[str]) -> Set[int]:
    ids = set()
    for t in toks:
        if hasattr(vocab, "token2id") and t in vocab.token2id:
            ids.add(vocab.token2id[t])
        elif hasattr(vocab, "stoi") and t in vocab.stoi:
            ids.add(vocab.stoi[t])
        else:
            # 若 vocab 里没有该 token，忽略即可（不同词表版本常见）
            pass
    return ids


def build_token_weight_vector(
    vocab: Vocab,
    *,
    w_digit: float = 1.8,
    w_var: float = 1.3,
    w_struct: float = 1.6,
    w_unit: float = 1.4,
) -> torch.Tensor:
    """
    返回 shape=(V,) 的 weight 向量，供 F.cross_entropy(weight=...) 使用。
    - 数字相关：0-9, '.', '-'  （抑制 4 . 5 -> 45、5x->6x 等）
    - 变量/希腊字母：a-zA-Z, 常用 \\alpha \\beta \\pi ...
    - 结构 token：{ } ( ) [ ]  ^ _  \\frac \\sqrt \\left \\right \\begin \\end ...
    - 单位/符号：cm, kg, Pa, \\Omega ...
    """
    V = len(vocab)
    w = torch.ones(V, dtype=torch.float32)

    # --- 1) 数字/小数点/负号 ---
    digit_tokens = [str(i) for i in range(10)] + [".", "-"]
    digit_ids = _ids_for_tokens(vocab, digit_tokens)
    for i in digit_ids:
        w[i] = w_digit

    # --- 2) 变量（字母）+ 常见希腊字母 ---
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    greek = [
        r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\theta",
        r"\lambda", r"\mu", r"\pi", r"\rho", r"\sigma", r"\omega", r"\Omega", r"\Delta", r"\Lambda",
        r"\varphi", r"\phi", r"\xi", r"\eta", r"\zeta",
    ]
    var_ids = _ids_for_tokens(vocab, letters + greek)
    for i in var_ids:
        w[i] = max(float(w[i]), w_var)

    # --- 3) 结构 token（强烈建议加权） ---
    struct = [
        "{", "}", "(", ")", "[", "]", "^", "_",
        r"\frac", r"\sqrt", r"\left", r"\right", r"\begin", r"\end",
        r"\overline", r"\boxed", r"\widehat",
        r"\cdot", r"\times", r"\div",
        r"\leq", r"\geq", r"\leqslant", r"\geqslant",
        r"\rightarrow", r"\Rightarrow", r"\Leftarrow", r"\Longrightarrow",
        r"\vert", "|", r"\mid", r"\parallel",
    ]
    struct_ids = _ids_for_tokens(vocab, struct)
    for i in struct_ids:
        w[i] = max(float(w[i]), w_struct)

    # --- 4) 常见单位/物理符号（可选） ---
    unit = ["cm", "m", "mm", "kg", "g", "N", "Pa", "A", "V", "W", r"\Omega"]
    unit_ids = _ids_for_tokens(vocab, unit)
    for i in unit_ids:
        w[i] = max(float(w[i]), w_unit)

    # PAD 不参与 loss（ignore_index 也会忽略，但把权重设 0 更保险）
    w[getattr(vocab, "pad_id", 0)] = 0.0
    return w


def weighted_cross_entropy(
    logits: torch.Tensor,         # (B,L,V)
    targets: torch.Tensor,        # (B,L)
    *,
    pad_id: int,
    weight_vec: torch.Tensor,     # (V,)
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    B, L, V = logits.shape
    weight_vec = weight_vec.to(logits.device)
    return torch.nn.functional.cross_entropy(
        logits.view(B * L, V),
        targets.view(B * L),
        ignore_index=pad_id,
        weight=weight_vec,
        label_smoothing=label_smoothing,
    )
