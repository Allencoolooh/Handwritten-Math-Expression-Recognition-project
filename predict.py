# predict.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import math
import torch
from PIL import Image
import numpy as np

from config import Config
from utils.vocab import Vocab
from model.model import MathFormulaRecognizer


# --------------------------------------------------------------- #
#                 图像预处理：与 Dataset 保持一致                  #
# --------------------------------------------------------------- #

def load_and_preprocess_image(
    img_path: Path,
    img_height: int,
    max_width: int,
) -> torch.Tensor:
    """
    加载单张图片，并做与 MathExprDataset 相同的预处理：
    - 灰度化
    - 等比例缩放到固定高度 img_height
    - 若宽度超过 max_width 则截断
    - 粘贴到 (max_width, img_height) 的黑色画布左上角
    - 转为 Tensor，shape = (1, H, W), 归一化到 [0,1]
    """
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # 1. 读图 & 灰度化
    img = Image.open(str(img_path)).convert("L")   # "L" 单通道灰度

    # 2. 等比例缩放
    orig_w, orig_h = img.size  # Pillow (W, H)
    scale = img_height / float(orig_h)
    new_w = int(math.ceil(orig_w * scale))
    if new_w > max_width:
        new_w = max_width

    img = img.resize((new_w, img_height), Image.BILINEAR)

    # 3. 右侧补零
    padded = Image.new("L", (max_width, img_height), color=0)
    padded.paste(img, (0, 0))

    # 4. 转 Tensor
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
        description="Handwritten Math Expression Recognition - Predict",
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
