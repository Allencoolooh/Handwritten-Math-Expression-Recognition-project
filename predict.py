# predict.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import math
import torch
from PIL import Image, ImageOps
import numpy as np

from config import Config
from utils.vocab import Vocab
from model.model import MathFormulaRecognizer


# --------------------------------------------------------------- #
#          一些辅助函数：自适应二值化 & 自动裁剪公式区域           #
# --------------------------------------------------------------- #

def _auto_binarize_and_invert(img: Image.Image) -> Image.Image:
    """
    对灰度图做一个简单的自适应阈值 + 可能的反色处理：
    - 先 autocontrast 提高对比度；
    - 用 mean/std 估一个阈值，把“字迹”拉成深色、背景拉成浅色；
    - 如果检测到是“黑底白字”，则反色变成白底黑字。
    """
    # 1. 自动对比度
    img = ImageOps.autocontrast(img)

    arr = np.array(img, dtype=np.uint8)  # (H, W), 0~255
    # 2. 粗略估计阈值：mean - k * std
    mean = arr.mean()
    std = arr.std()
    thr = mean - 0.3 * std  # k=0.3 可以稍微调

    # 文本像素：比阈值更“深”的部分
    text_mask = arr < thr
    bg_mask = ~text_mask

    # 如果几乎没有检测到文本，就直接返回原图（避免空掏）
    if text_mask.sum() < 10:
        return img

    # 判断是不是黑底白字：
    # - 文本区域平均值比背景区域平均值还“亮”很多 → 说明是白字黑底
    text_mean = arr[text_mask].mean()
    bg_mean = arr[bg_mask].mean()
    # 黑底白字：背景更暗，文字更亮
    if text_mean > bg_mean:
        # 反色
        img = ImageOps.invert(img)
        arr = np.array(img, dtype=np.uint8)
        # 重新计算掩码
        mean = arr.mean()
        std = arr.std()
        thr = mean - 0.3 * std
        text_mask = arr < thr
        if text_mask.sum() < 10:
            # 防止反色后掩码反而消失
            return img

    return img


def _crop_to_content(img: Image.Image, margin_ratio: float = 0.05) -> Image.Image:
    """
    根据非背景像素自动裁剪公式区域，去掉大块空白/背景。
    - 先做一个粗略阈值分割，得到“有字”的区域；
    - 找到最小外接矩形 bbox；
    - 四周再保留一点 margin。
    """
    arr = np.array(img, dtype=np.uint8)  # (H, W)
    H, W = arr.shape

    # 用整体直方图 + otsu/近似阈值会更好，但这里先用简单方案：
    mean = arr.mean()
    std = arr.std()
    thr = mean - 0.2 * std
    text_mask = arr < thr

    # 若基本全白或全黑，直接不裁剪
    ys, xs = np.where(text_mask)
    if len(xs) == 0 or len(ys) == 0:
        return img

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 加一点边距
    margin_x = int((x_max - x_min + 1) * margin_ratio)
    margin_y = int((y_max - y_min + 1) * margin_ratio)

    x_min = max(0, x_min - margin_x)
    x_max = min(W - 1, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(H - 1, y_max + margin_y)

    # 防止裁剪得过小
    if x_max <= x_min or y_max <= y_min:
        return img

    cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped


# --------------------------------------------------------------- #
#                 图像预处理：适配拍照/扫描图片                     #
# --------------------------------------------------------------- #

def load_and_preprocess_image(
    img_path: Path,
    img_height: int,
    max_width: int,
) -> torch.Tensor:
    """
    加载单张图片，并做“适配拍照图片”的预处理：
    1. 转灰度；
    2. 自动对比度 + 近似二值化 + 必要时反色（确保白底黑字更常见）；
    3. 自动裁剪公式区域（去掉大块空白和背景）；
    4. 等比例缩放到固定高度 img_height；
    5. 若宽度超过 max_width 则截断；
    6. 粘贴到 (max_width, img_height) 的黑色画布左上角；
    7. 转为 Tensor，shape = (1, H, W), 归一化到 [0,1]。

    注意：训练时的 Dataset 没有 2/3 步，这里相当于给“用户上传图片”加一个适配器，
    尽量把它们调理成和训练图片更类似的风格。
    """
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # 1. 读图 & 灰度化
    img = Image.open(str(img_path)).convert("L")   # "L" 单通道灰度

    # 2. 对比度增强 + 近似二值化 + 必要时反色
    img = _auto_binarize_and_invert(img)

    # 3. 自动裁剪到“有字”的区域
    img = _crop_to_content(img)

    # —— 从这一步开始，和训练时的 resize+pad 更接近 ——

    # 4. 等比例缩放
    orig_w, orig_h = img.size  # Pillow (W, H)
    scale = img_height / float(orig_h)
    new_w = int(math.ceil(orig_w * scale))
    if new_w > max_width:
        new_w = max_width

    img = img.resize((new_w, img_height), Image.BILINEAR)

    # 5. 右侧补零
    padded = Image.new("L", (max_width, img_height), color=0)
    padded.paste(img, (0, 0))

    # 6. 转 Tensor
    arr = np.array(padded, dtype=np.uint8)   # (H, W)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.unsqueeze(0)            # (1, H, W)

    return tensor


# --------------------------------------------------------------- #
#                 加载模型权重（best.pt）                         #
# --------------------------------------------------------------- #

def load_model_and_vocab() -> tuple[MathFormulaRecognizer, Vocab]:
    """
    加载 vocab + 模型，并从 Config.CKPT_DIR / best.pt 恢复权重。
    """
    # 1. 加载 vocab
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    print(f"[Predict] Vocab size: {len(vocab)}")

    # 2. 构建模型
    model = MathFormulaRecognizer(vocab).to(Config.DEVICE)

    # 3. 加载 best checkpoint
    ckpt_dir = getattr(Config, "CKPT_DIR", Path("checkpoints"))
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best.pt"
    assert ckpt_path.is_file(), f"[Predict] best.pt not found at: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 兼容不同保存格式
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[Predict] Warning - missing keys in state_dict:", missing)
    if unexpected:
        print("[Predict] Warning - unexpected keys in state_dict:", unexpected)

    model.to(Config.DEVICE)
    model.eval()

    print(f"[Predict] Loaded best model from {ckpt_path}")
    return model, vocab


# --------------------------------------------------------------- #
#                          推理主函数                             #
# --------------------------------------------------------------- #

def predict_on_images(
    image_paths: List[Path],
    decode_method: str = "beam",
    beam_size: int = 3,
    max_len: int | None = None,
) -> None:
    """
    对一组图片路径进行预测，并在终端打印结果。
    decode_method: "beam" 或 "greedy"
    beam_size:     仅在 beam 模式下生效
    """
    if max_len is None:
        max_len = getattr(Config, "MAX_TGT_LEN", 128)

    # 加载模型和词表
    model, vocab = load_model_and_vocab()

    # 预处理所有图片，组成一个 batch
    imgs: List[torch.Tensor] = []
    valid_paths: List[Path] = []

    for p in image_paths:
        try:
            tensor = load_and_preprocess_image(
                p,
                img_height=Config.IMG_HEIGHT,
                max_width=Config.MAX_WIDTH,
            )
            imgs.append(tensor)
            valid_paths.append(p)
        except FileNotFoundError as e:
            print(f"[Predict] {e}")

    if not imgs:
        print("[Predict] No valid images to process.")
        return

    images_batch = torch.stack(imgs, dim=0).to(Config.DEVICE)  # (B, 1, H, W)

    # 调用模型进行解码
    if decode_method == "beam":
        print(f"[Predict] Using beam search decode, beam_size={beam_size}")
        preds = model.recognize_beam(
            images_batch,
            max_len=max_len,
            device=Config.DEVICE,
            beam_size=beam_size,
        )
    else:
        print("[Predict] Using greedy decode")
        preds = model.recognize(
            images_batch,
            max_len=max_len,
            device=Config.DEVICE,
        )

    # 打印结果
    print("\n========== Prediction Results ==========\n")
    for path, latex in zip(valid_paths, preds):
        print(f"[Image] {path}")
        print(f"[LaTeX] {latex}")
        print("-" * 60)


# --------------------------------------------------------------- #
#                          命令行入口                             #
# --------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Handwritten Math Expression Recognition - Predict (photo-adaptive)",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="image file paths to predict",
    )
    parser.add_argument(
        "--decode",
        choices=["beam", "greedy"],
        default="beam",
        help="decode method: beam or greedy (default: beam)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="beam size for beam search (default: 3)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="max decoding length, default: Config.MAX_TGT_LEN",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = [Path(p) for p in args.images]

    predict_on_images(
        image_paths=image_paths,
        decode_method=args.decode,
        beam_size=args.beam_size,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
