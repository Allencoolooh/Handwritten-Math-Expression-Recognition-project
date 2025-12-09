# inference.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms as T

from config import Config
from utils.vocab import Vocab
from model.model import MathFormulaRecognizer


# ----------------- 全局初始化：设备 / 词表 / 模型 / transform -----------------
_device = Config.DEVICE  # 里面已经是 cuda 或 cpu
_vocab = Vocab.from_file(Path(Config.VOCAB_PATH))

_model = MathFormulaRecognizer(vocab=_vocab).to(_device)
_model.eval()

# 默认 checkpoint 路径（你可以改成自己实际的 best.pt 路径）
_default_ckpt_path = Path(Config.CKPT_DIR) / "best.pt"
# 如果你训练时保存在 results/checkpoints 下，也可以改成：
# _default_ckpt_path = Config.CHECKPOINT_DIR / "best.pt"

# 加载权重
_state = torch.load(_default_ckpt_path, map_location=_device)
if isinstance(_state, dict):
    if "model" in _state:
        _model.load_state_dict(_state["model"], strict=False)
    elif "state_dict" in _state:
        _model.load_state_dict(_state["state_dict"], strict=False)
    else:
        _model.load_state_dict(_state, strict=False)
else:
    _model.load_state_dict(_state, strict=False)

print(f"[inference] Loaded checkpoint from: {_default_ckpt_path}")

_transform = T.Compose([
    T.Resize((Config.IMG_HEIGHT, Config.MAX_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


# ----------------- 对外暴露的预测函数 -----------------
def predict_latex_from_pil(
    img: Image.Image,
    use_beam: bool = False,
    max_len: Optional[int] = None,
) -> str:
    """
    输入：PIL Image（任意大小）
    输出：识别出的 LaTeX 字符串
    """
    if max_len is None:
        max_len = Config.MAX_TGT_LEN

    img = img.convert("L")
    tensor = _transform(img).unsqueeze(0).to(_device)  # (1, 1, H, W)

    with torch.no_grad():
        if use_beam and hasattr(_model, "recognize_beam"):
            texts = _model.recognize_beam(
                images=tensor,
                max_len=max_len,
                device=_device,
                beam_size=3,
            )
        else:
            texts = _model.recognize(
                images=tensor,
                max_len=max_len,
                device=_device,
            )

    return texts[0] if len(texts) > 0 else ""


def predict_latex_from_path(
    img_path: str | Path,
    use_beam: bool = False,
    max_len: Optional[int] = None,
) -> str:
    """
    如果你有图片路径，也可以直接用这个函数。
    """
    img_path = Path(img_path)
    img = Image.open(img_path)
    return predict_latex_from_pil(img, use_beam=use_beam, max_len=max_len)
