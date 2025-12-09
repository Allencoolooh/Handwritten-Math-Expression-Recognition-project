# tools/split_dataset.py
from pathlib import Path
import random

from config import Config


def main():
    # 1. 完整标注文件路径（根据你实际情况修改）
    #    假设你之前有 data/train_full_icdar.txt
    full_path = Config.DATA_DIR / "train_full_icdar.txt"

    assert full_path.is_file(), f"Full label file not found: {full_path}"

    with full_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    print(f"[Split] Total samples: {len(lines)}")

    # 2. 打乱
    random.seed(42)
    random.shuffle(lines)

    n = len(lines)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    print(f"[Split] Train: {len(train_lines)}, Val: {len(val_lines)}, Test: {len(test_lines)}")

    # 3. 写回 data/train.txt, data/val.txt, data/test.txt
    out_train = Config.DATA_DIR / "train.txt"
    out_val = Config.DATA_DIR / "val.txt"
    out_test = Config.DATA_DIR / "test.txt"

    def write_lines(path: Path, lines_):
        with path.open("w", encoding="utf-8") as f:
            for line in lines_:
                f.write(line + "\n")
        print(f"[Split] Wrote {len(lines_)} lines to {path}")

    write_lines(out_train, train_lines)
    write_lines(out_val, val_lines)
    write_lines(out_test, test_lines)


if __name__ == "__main__":
    main()
