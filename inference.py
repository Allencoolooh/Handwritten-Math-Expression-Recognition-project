from __future__ import annotations

from pathlib import Path
from typing import Optional

import math
import numpy as np
import torch
from PIL import Image

from config import Config
from utils.vocab import Vocab
from model.model import MathFormulaRecognizer


# --------------------------------------------------------------- #
#                 图像预处理：与 Dataset 保持一致                  #
# --------------------------------------------------------------- #

def preprocess_pil_image(
    img: Image.Image,
    img_height: int,
    max_width: int,
) -> torch.Tensor:
    """
    针对 PIL Image 做与 MathExprDataset 相同的预处理：
    - 灰度化
    - 等比例缩放到固定高度 img_height
    - 若宽度超过 max_width 则截断
    - 粘贴到 (max_width, img_height) 的黑色画布左上角
    - 转为 Tensor，shape = (1, H, W)，归一化到 [0,1]
    """
    img = img.convert("L")  # 灰度

    # 等比例缩放
    orig_w, orig_h = img.size  # (W, H)
    scale = img_height / float(orig_h)
    new_w = int(math.ceil(orig_w * scale))
    if new_w > max_width:
        new_w = max_width

    img = img.resize((new_w, img_height), Image.BILINEAR)

    # 右侧补零
    padded = Image.new("L", (max_width, img_height), color=0)
    padded.paste(img, (0, 0))

    # 转 tensor
    arr = np.array(padded, dtype=np.uint8)  # (H, W)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1, H, W)

    return tensor


# --------------------------------------------------------------- #
#               全局：加载 vocab + 模型 + 权重一次                 #
# --------------------------------------------------------------- #

_device = Config.DEVICE

# 1. 词表
_vocab = Vocab.from_file(Config.VOCAB_PATH)
print(f"[Inference] Vocab size: {len(_vocab)}")

# 2. 模型
_model = MathFormulaRecognizer(_vocab).to(_device)

# 3. 权重：沿用 predict.py 的逻辑，从 Config.CKPT_DIR / best.pt 加载
_ckpt_dir = getattr(Config, "CKPT_DIR", Path("checkpoints"))
if not isinstance(_ckpt_dir, Path):
    _ckpt_dir = Path(_ckpt_dir)
_ckpt_path = _ckpt_dir / "best.pt"
assert _ckpt_path.is_file(), f"[Inference] best.pt not found at: {_ckpt_path}"

_ckpt = torch.load(_ckpt_path, map_location="cpu")

if "model_state" in _ckpt:
    _state_dict = _ckpt["model_state"]
elif "model" in _ckpt:
    _state_dict = _ckpt["model"]
else:
    _state_dict = _ckpt

_missing, _unexpected = _model.load_state_dict(_state_dict, strict=False)
if _missing:
    print("[Inference] Warning - missing keys in state_dict:", _missing)
if _unexpected:
    print("[Inference] Warning - unexpected keys in state_dict:", _unexpected)

_model.to(_device)
_model.eval()

print(f"[Inference] Loaded best model from {_ckpt_path}")


# --------------------------------------------------------------- #
#                    对外提供的预测函数接口                        #
# --------------------------------------------------------------- #

def predict_latex_from_pil(
    img: Image.Image,
    decode_method: str = "beam",
    beam_size: int = 3,
    max_len: Optional[int] = None,
) -> str:
    """
    输入：PIL Image
    输出：LaTeX 字符串
    decode_method: "beam" 或 "greedy"
    """
    if max_len is None:
        max_len = getattr(Config, "MAX_TGT_LEN", 128)

    # 预处理，与训练数据保持一致
    tensor = preprocess_pil_image(
        img,
        img_height=Config.IMG_HEIGHT,
        max_width=Config.MAX_WIDTH,
    ).unsqueeze(0).to(_device)  # (1, 1, H, W)

    with torch.no_grad():
        if decode_method == "beam" and hasattr(_model, "recognize_beam"):
            preds = _model.recognize_beam(
                images=tensor,
                max_len=max_len,
                device=_device,
                beam_size=beam_size,
            )
        else:
            preds = _model.recognize(
                images=tensor,
                max_len=max_len,
                device=_device,
            )

    if not preds:
        return ""
    return preds[0]


def predict_latex_from_path(
    img_path: str | Path,
    decode_method: str = "beam",
    beam_size: int = 3,
    max_len: Optional[int] = None,
) -> str:
    """
    方便命令行/脚本直接用图片路径调用。
    """
    img_path = Path(img_path)
    img = Image.open(str(img_path))
    return predict_latex_from_pil(
        img,
        decode_method=decode_method,
        beam_size=beam_size,
        max_len=max_len,
    )
