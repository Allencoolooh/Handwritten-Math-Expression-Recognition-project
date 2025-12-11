# tools/build_oversampled_train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_line(line: str) -> Tuple[str, str] | None:
    """
    解析一行标注：
        img_rel\tlatex
    返回 (img_rel, latex) 或 None（非法行）。
    """
    line = line.strip()
    if not line:
        return None
    try:
        img_rel, latex = line.split("\t", 1)
    except ValueError:
        return None

    img_rel = img_rel.strip()
    latex = latex.strip()
    if not img_rel or not latex:
        return None
    return img_rel, latex


def latex_len_tokens(latex: str) -> int:
    """
    用空格分词估计 LaTeX token 长度。
    你的标注已经是 token 间以空格分隔，这个估计较准确。
    """
    return len(latex.strip().split()) if latex.strip() else 0


def oversample_train(
    src: Path,
    dst: Path,
    thr_short: int,
    thr_mid: int,
    factor_short: int,
    factor_mid: int,
    factor_long: int,
) -> None:
    """
    根据长度对公式做 oversampling 并写入新文件 dst。

    三段划分：
      - L <= thr_short            → 短公式，重复 factor_short 次
      - thr_short < L <= thr_mid  → 中公式，重复 factor_mid 次
      - L > thr_mid               → 长公式，重复 factor_long 次

    一般建议：
      thr_short = 10, thr_mid = 20
      factor_short = 1, factor_mid = 1, factor_long = 3~4
    """
    assert src.is_file(), f"Source labels file not found: {src}"

    all_count = 0
    short_count = mid_count = long_count = 0
    out_lines: List[str] = []

    with src.open("r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_line(raw)
            if parsed is None:
                continue

            img_rel, latex = parsed
            L = latex_len_tokens(latex)
            all_count += 1

            # 确定当前样本所属长度段和重复次数
            if L <= thr_short:
                bucket = "short"
                repeat = factor_short
                short_count += 1
            elif L <= thr_mid:
                bucket = "mid"
                repeat = factor_mid
                mid_count += 1
            else:
                bucket = "long"
                repeat = factor_long
                long_count += 1

            for _ in range(repeat):
                out_lines.append(f"{img_rel}\t{latex}\n")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print("====== Oversample Summary ======")
    print(f"Source file      : {src}")
    print(f"Output file      : {dst}")
    print(f"Original samples : {all_count}")
    print(f"  short (<= {thr_short:2d})     : {short_count}")
    print(f"  mid   ({thr_short:2d} < L <= {thr_mid:2d}) : {mid_count}")
    print(f"  long  (> {thr_mid:2d})       : {long_count}")
    print("--------------------------------")
    print(f"Output lines     : {len(out_lines)}")
    print(f"Repeat factors   : short={factor_short}, mid={factor_mid}, long={factor_long}")
    print("================================")


def main():
    parser = argparse.ArgumentParser(
        description="Oversample formulas of different lengths in train labels."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="data/train.txt",
        help="原始训练标注文件路径 (默认: data/train.txt)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/train_oversampled.txt",
        help="输出 oversample 后的标注文件路径 (默认: data/train_oversampled.txt)",
    )

    # 长度阈值
    parser.add_argument(
        "--thr-short",
        type=int,
        default=10,
        help="短公式阈值: L <= thr_short 认为是短 (默认: 10)",
    )
    parser.add_argument(
        "--thr-mid",
        type=int,
        default=20,
        help="中公式阈值: thr_short < L <= thr_mid 认为是中 (默认: 20)",
    )

    # 各段重复次数
    parser.add_argument(
        "--factor-short",
        type=int,
        default=1,
        help="短公式重复次数 (默认: 1)",
    )
    parser.add_argument(
        "--factor-mid",
        type=int,
        default=1,
        help="中公式重复次数 (默认: 1)",
    )
    parser.add_argument(
        "--factor-long",
        type=int,
        default=3,
        help="长公式重复次数 (默认: 3，建议 3~4)",
    )

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    oversample_train(
        src=src,
        dst=dst,
        thr_short=args.thr_short,
        thr_mid=args.thr_mid,
        factor_short=args.factor_short,
        factor_mid=args.factor_mid,
        factor_long=args.factor_long,
    )


if __name__ == "__main__":
    main()
