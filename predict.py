#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单张图片预测脚本（完全匹配当前仓库实现）：

示例：
    python predict.py --img data/test_images/0001.png --ckpt checkpoints/best.pt

说明：
    - 使用 MathFormulaRecognizer 中已经实现好的 recognize() / recognize_beam()
    - 图像大小使用 Config.IMG_HEIGHT / Config.MAX_WIDTH
    - 词表路径使用 Config.VOCAB_PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from config import Config
from utils.vocab import Vocab
from model.model import MathFormulaRecognizer


# ---------------------------------------------------------
# 图片预处理：灰度 + Resize 到 (IMG_HEIGHT, MAX_WIDTH) + ToTensor + 归一化
# ---------------------------------------------------------
def load_image(img_path: str | Path) -> torch.Tensor:
    """
    读取并预处理图片，返回形状为 (1, 1, H, W) 的张量。
    H = Config.IMG_HEIGHT, W = Config.MAX_WIDTH
    """
    img_path = Path(img_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("L")  # 转灰度

    transform = T.Compose([
        T.Resize((Config.IMG_HEIGHT, Config.MAX_WIDTH)),
        T.ToTensor(),                     # (1, H, W), 值域 [0,1]
        T.Normalize(mean=[0.5], std=[0.5])  # 简单归一化到大致 [-1,1]
    ])

    tensor = transform(img)          # (1, H, W)
    tensor = tensor.unsqueeze(0)     # (1, 1, H, W) —— batch_size = 1
    return tensor


# ---------------------------------------------------------
# 构建 vocab 和 模型（严格按照你现在的 __init__ 签名）
# ---------------------------------------------------------
def build_model_and_vocab(device: torch.device):
    """
    - 从 Config.VOCAB_PATH 加载 Vocab
    - 用 Vocab 初始化 MathFormulaRecognizer(vocab=vocab)
    """
    vocab_path = Path(Config.VOCAB_PATH)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    vocab = Vocab.from_file(vocab_path)

    # 你的 MathFormulaRecognizer __init__:
    #   def __init__(self, vocab, d_model=Config.D_MODEL, ...)
    # 这些超参都已经从 Config 取默认值了，所以这里只传 vocab 即可。
    model = MathFormulaRecognizer(vocab=vocab)
    model.to(device)

    return model, vocab


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device):
    """
    加载训练好的权重，兼容几种常见保存方式：
    - torch.save(model.state_dict(), path)
    - torch.save({"model": model.state_dict(), ...}, path)
    - torch.save({"state_dict": model.state_dict(), ...}, path)
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            # 可能直接就是 state_dict
            model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    print(f"✅ Loaded checkpoint from: {ckpt_path}")


# ---------------------------------------------------------
# 主函数：调用 model.recognize() / recognize_beam() 完成解码
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict LaTeX from a handwritten math expression image.")
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=Config.MAX_TGT_LEN,
        help="Maximum decoding length (传给 model.recognize / recognize_beam).",
    )
    parser.add_argument(
        "--beam",
        action="store_true",
        help="是否使用 beam search（调用 model.recognize_beam），默认用 greedy。",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1) 构建模型和 vocab
    model, vocab = build_model_and_vocab(device)

    # 2) 加载 checkpoint
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    # 3) 读入图片并预处理
    image = load_image(args.img).to(device)  # (1, 1, H, W)

    # 4) 解码（使用你在 model 里已经写好的接口）
    with torch.no_grad():
        if args.beam and hasattr(model, "recognize_beam"):
            texts = model.recognize_beam(
                images=image,
                max_len=args.max_len,
                device=device,
                beam_size=3,  # 如需可调，可以再加 argparse 参数
            )
        else:
            texts = model.recognize(
                images=image,
                max_len=args.max_len,
                device=device,
            )

    # 5) 输出结果（单张图，所以 texts[0]）
    pred_latex = texts[0] if len(texts) > 0 else ""

    print("\n===== 预测结果（LaTeX） =====")
    print(pred_latex)
    print("================================\n")


if __name__ == "__main__":
    main()
