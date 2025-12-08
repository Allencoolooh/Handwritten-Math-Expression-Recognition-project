import os
import random

#############################################################
#（1） 文件名 + LaTeX 分割函数 —— 根据你的真实数据格式定制
#############################################################

def split_filename_and_latex(line: str):
    line = line.strip()
    if not line:
        return None, None

    # 数据集是 .jpg 图片
    exts = [".jpg", ".png", ".jpeg", ".bmp"]
    end_idx = None

    for ext in exts:
        idx = line.find(ext)
        if idx != -1:
            end_idx = idx + len(ext)
            break

    if end_idx is None:
        print(f"[WARN] Cannot parse line: {line[:50]}")
        return None, None

    filename = line[:end_idx]
    latex = line[end_idx:].lstrip()

    return filename, latex


#############################################################
#（2） 转换：原始 ICDAR 格式 → 我们项目格式
#############################################################

def convert_icdar_to_our_format(src_labels, dst_file, image_prefix):
    total = 0
    ok = 0

    with open(src_labels, "r", encoding="utf-8") as fin, \
         open(dst_file, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            filename, latex = split_filename_and_latex(line)
            if filename is None:
                continue

            # 组合成我们项目的图片路径（相对根目录）
            img_path = os.path.join(image_prefix, filename)

            # 写入统一格式：图片路径<TAB>latex
            fout.write(f"{img_path}\t{latex}\n")
            ok += 1

    print(f"[INFO] Total={total}, Converted={ok}")


#############################################################
#（3） 划分 train/val
#############################################################

def split_train_val(full_file, train_file, val_file, val_ratio=0.1, seed=42):
    random.seed(seed)

    with open(full_file, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    random.shuffle(lines)

    n_total = len(lines)
    n_val = int(n_total * val_ratio)

    val_lines = lines[:n_val]
    train_lines = lines[n_val:]

    with open(train_file, "w", encoding="utf-8") as ftrain:
        ftrain.writelines(train_lines)

    with open(val_file, "w", encoding="utf-8") as fval:
        fval.writelines(val_lines)

    print(f"[INFO] Train={len(train_lines)}, Val={len(val_lines)}")


#############################################################
#（4） 主流程（你在项目根目录运行此脚本）
#############################################################

if __name__ == "__main__":

    # 原始标注路径（从根目录出发）
    SRC_LABELS = "data/icdar_raw/train_labels.txt"

    # 转换后的完整标注
    FULL_OUT = "data/train_full_icdar.txt"

    # 图片所在目录（从根目录出发）
    IMAGE_PREFIX = "data/icdar_raw/train_images"

    # Step 1：转换格式
    convert_icdar_to_our_format(
        src_labels=SRC_LABELS,
        dst_file=FULL_OUT,
        image_prefix=IMAGE_PREFIX
    )

    # Step 2：划分 train/val
    split_train_val(
        full_file=FULL_OUT,
        train_file="data/train.txt",
        val_file="data/val.txt",
        val_ratio=0.1,
        seed=42
    )
