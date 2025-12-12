from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


def parse_line(line: str) -> Tuple[str, str] | None:
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
    return len(latex.strip().split()) if latex.strip() else 0


def main():
    parser = argparse.ArgumentParser("Build long-formula fine-tune training file.")
    parser.add_argument("--src", type=str, default="data/train_oversampled.txt")
    parser.add_argument("--dst", type=str, default="data/train_long.txt")
    parser.add_argument("--min-len", type=int, default=21, help="Keep samples with L >= min-len")
    parser.add_argument("--repeat-very-long", type=int, default=2, help="Repeat very long samples")
    parser.add_argument("--very-long-len", type=int, default=31, help="Very long threshold")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    assert src.is_file(), f"Source file not found: {src}"

    kept = 0
    written = 0
    very_long = 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            parsed = parse_line(raw)
            if parsed is None:
                continue
            img_rel, latex = parsed
            L = latex_len_tokens(latex)
            if L < args.min_len:
                continue

            kept += 1
            rep = 1
            if L >= args.very_long_len:
                rep = max(1, int(args.repeat_very_long))
                very_long += 1

            for _ in range(rep):
                fout.write(f"{img_rel}\t{latex}\n")
                written += 1

    print("===== Build Long Train Summary =====")
    print(f"src            : {src}")
    print(f"dst            : {dst}")
    print(f"min_len        : {args.min_len}")
    print(f"very_long_len  : {args.very_long_len}")
    print(f"repeat_very_long: {args.repeat_very_long}")
    print("------------------------------------")
    print(f"kept samples   : {kept}")
    print(f"very long cnt  : {very_long}")
    print(f"written lines  : {written}")
    print("====================================")


if __name__ == "__main__":
    main()
