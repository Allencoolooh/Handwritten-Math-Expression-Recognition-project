# utils/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import math

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from .vocab import Vocab
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
)


class MathExprDataset(Dataset):
    """
    手写数学表达式识别数据集封装。

    功能：
    - 从标注文件中读取 (图片相对路径, LaTeX 字符串)
    - 加载图片 -> 灰度化 -> 等比例缩放到固定高度 -> 右侧补零到固定宽度
    - 调用 Vocab 将 LaTeX 文本编码成 id 序列（含 <SOS>/<EOS>）
    """

    def __init__(
        self,
        labels_path: Path,
        vocab: Vocab,
        img_base_dir: Path = Config.IMG_BASE_DIR,
        img_height: int = Config.IMG_HEIGHT,
        max_width: int = Config.MAX_WIDTH,
        augment: bool = False,
    ) -> None:
        """
        参数：
        - labels_path : 标注文件路径，如 data/train.txt
        - vocab       : 已经构建好的 Vocab 实例
        - img_base_dir: 图片根目录（标注文件里是相对路径，会拼在它后面）
        - img_height  : 统一缩放后的高度（像素）
        - max_width   : 统一的最大宽度（像素），不足右侧补零，超出裁剪
        - augment     : 是否进行数据增强（当前预留接口，先不做）
        """
        super().__init__()
        self.labels_path = Path(labels_path)
        self.vocab = vocab
        self.img_base_dir = Path(img_base_dir)
        self.img_height = img_height
        self.max_width = max_width
        self.augment = augment

        assert self.labels_path.is_file(), f"Labels file not found: {self.labels_path}"

        # 解析标注文件，得到 (img_rel_path, latex_str) 列表
        self.samples: List[Tuple[str, str]] = self._load_samples(self.labels_path)

    # ------------------------------------------------------------------ #
    #                1. 读取标注文件：img_path + latex                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_samples(labels_path: Path) -> List[Tuple[str, str]]:
        """
        从标注文件读取所有样本。

        标注文件格式约定：
            每行:  图片相对路径 \t LaTeX字符串
        """
        samples: List[Tuple[str, str]] = []

        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 尝试按制表符拆成两部分：路径 和 LaTeX
                try:
                    img_rel, latex = line.split("\t", 1)
                except ValueError:
                    # 非法行，直接跳过
                    continue

                img_rel = img_rel.strip()
                latex = latex.strip()
                if img_rel == "" or latex == "":
                    continue

                samples.append((img_rel, latex))

        print(f"[Dataset] Loaded {len(samples)} samples from {labels_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    #                     2. 图像加载 & 预处理                           #
    # ------------------------------------------------------------------ #
    def _load_image(self, img_path: Path) -> Image.Image:
        """
        加载图片并转为灰度图 (mode='L')。
        """
        img = Image.open(img_path).convert("L")  # 'L' 表示单通道灰度
        return img

    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        """
        预留的数据增强接口，目前不做任何修改，直接返回。

        你以后可以在这里加：
        - 轻微旋转 / 平移
        - 轻微缩放 / 仿射变换
        - 加一点噪声 / 模糊 等
        """
        return img

    def _resize_and_pad(self, img: Image.Image) -> torch.Tensor:
        """
        图像预处理流程：

        1. 等比例缩放到固定高度 self.img_height；
        2. 根据比例计算新宽度 new_w，若 new_w > max_width 则截断为 max_width；
        3. 将缩放后的图像粘贴到 (max_width, img_height) 的全黑画布左上角；
        4. 转成 float32 的 Tensor，形状为 (1, H, W)，数值范围 [0, 1]。
        """
        orig_w, orig_h = img.size  # Pillow 的尺寸格式是 (W, H)

        # 1. 计算缩放比例，使得高度变为 img_height
        scale = self.img_height / float(orig_h)
        new_w = int(math.ceil(orig_w * scale))

        # 避免过长：限制最大宽度
        if new_w > self.max_width:
            new_w = self.max_width

        # 2. 按新尺寸缩放
        img = img.resize((new_w, self.img_height), Image.BILINEAR)

        # 3. 新建 (max_width, img_height) 的全黑图像，并将缩放后的图粘贴到左上角
        padded = Image.new("L", (self.max_width, self.img_height), color=0)
        padded.paste(img, (0, 0))

        # 4. 转 numpy -> Tensor，注意 shape: (H, W) -> (1, H, W)
        arr = np.array(padded, dtype=np.uint8)  # (H, W), 值在 [0, 255]
        tensor = torch.from_numpy(arr).float() / 255.0  # 归一化到 [0,1]
        tensor = tensor.unsqueeze(0)  # (1, H, W)

        return tensor

    # ------------------------------------------------------------------ #
    #                          3. __getitem__                            #
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回一个样本字典：
            {
                "image": (1, H, W) 张量，float32，[0,1]
                "label": 原始 LaTeX 字符串
                "tgt_ids": LongTensor (T,) 编码后的 token 序列（含 <SOS>/<EOS>）
            }
        """
        img_rel, latex = self.samples[idx]

        img_path = self.img_base_dir / img_rel
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 1. 加载原图
        img = self._load_image(img_path)

        # 2. 数据增强（目前为空操作）
        if self.augment:
            img = self._apply_augmentation(img)

        # 3. resize + pad + 转 Tensor
        img_tensor = self._resize_and_pad(img)  # (1, H, W)

        # 4. 文本编码为 id 序列（内部用的是新版 LaTeX tokenizer）
        tgt_ids = self.vocab.encode(latex, add_special_tokens=True)  # List[int]
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)           # (T,)

        sample = {
            "image": img_tensor,
            "label": latex,
            "tgt_ids": tgt_ids,
        }
        return sample


class RandomIndexSampler(Sampler[int]):
    """
    一个极简随机采样器：
    - 每次迭代时，返回 [0, 1, ..., len(dataset)-1] 的一个随机排列。
    - 不依赖 torch 内置 RandomSampler，所以不会触发 num_samples 的 bug。
    """

    def __init__(self, data_source, generator: torch.Generator | None = None):
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            # 用全局随机数生成器
            indices = torch.randperm(n).tolist()
        else:
            # 用指定 generator（一般用不上）
            indices = torch.randperm(n, generator=self.generator).tolist()
        return iter(indices)

    def __len__(self):
        # 这很关键，BatchSampler 会用到 len(sampler)
        return len(self.data_source)


# ---------------------------------------------------------------------- #
#                 4. collate_fn：对一个 batch 做对齐                     #
# ---------------------------------------------------------------------- #
def math_collate_fn(
    batch: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """
    自定义的 collate_fn，给 DataLoader 使用。

    输入 batch: List[样本字典]，每个样本是 MathExprDataset.__getitem__ 的返回：
        {
            "image": (1, H, W),
            "label": str,
            "tgt_ids": (T_i,)
        }

    输出一个 batch 字典：
        {
            "images":   (B, 1, H, W)
            "tgt_input":  (B, L-1)   # decoder 输入序列
            "tgt_output": (B, L-1)   # 监督标签（预测下一个 token）
            "tgt_lengths": (B,)      # 有效长度（不含 padding，且对应 L-1）
            "labels":   List[str]    # 原始 LaTeX 文本
        }
    """
    # 1. 图像可以直接堆叠，因为都已被 resize + pad 为相同大小
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B, 1, H, W)

    # 2. 文本序列长度
    tgt_seqs: List[torch.Tensor] = [b["tgt_ids"] for b in batch]
    lengths = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)  # (B,)
    max_len = int(lengths.max().item())

    batch_size = len(batch)

    # 3. 用 pad_id 把所有序列 pad 到相同长度
    tgt_padded = torch.full(
        (batch_size, max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )  # (B, max_len)

    for i, seq in enumerate(tgt_seqs):
        L = len(seq)
        tgt_padded[i, :L] = seq

    # 4. 构造 decoder 的输入 / 输出序列
    #    假设原始目标序列是 [SOS, a, b, c, EOS]
    #    则：
    #      tgt_input  = [SOS, a, b, c]
    #      tgt_output = [a,   b, c, EOS]
    y_in = tgt_padded[:, :-1]   # (B, max_len-1)
    y_out = tgt_padded[:, 1:]   # (B, max_len-1)

    out_lengths = lengths - 1   # 每个样本有效长度也相应减 1

    batch_dict = {
        "images": images,          # (B, 1, H, W)
        "tgt_input": y_in,         # (B, L-1)
        "tgt_output": y_out,       # (B, L-1)
        "tgt_lengths": out_lengths,  # (B,)
        "labels": [b["label"] for b in batch],
    }
    return batch_dict


# ---------------------------------------------------------------------- #
#             5. 一个方便的 DataLoader 工厂函数                         #
# ---------------------------------------------------------------------- #
'''def create_dataloader(
    labels_path: Path,
    vocab: Vocab,
    batch_size: int,
    shuffle: bool,          # 这个参数先保留接口，但暂时不用
    num_workers: int = 0,
    augment: bool = False,
) -> DataLoader:
    """
    极简版 DataLoader：不使用 RandomSampler/BatchSampler，只用最基础的顺序取数。
    先验证 Dataset + collate_fn 全部工作正常，后面再加 shuffle。
    """
    dataset = MathExprDataset(
        labels_path=labels_path,
        vocab=vocab,
        img_base_dir=Config.IMG_BASE_DIR,
        img_height=Config.IMG_HEIGHT,
        max_width=Config.MAX_WIDTH,
        augment=augment,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,              # ★ 这里固定为 False，先别用内部 RandomSampler
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: math_collate_fn(batch, pad_id=vocab.pad_id),
    )

    return loader
'''


def create_dataloader(
    labels_path: Path,
    vocab: Vocab,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    augment: bool = False,
) -> DataLoader:
    """
    最终可用于训练的 DataLoader 构造函数：
    - shuffle=False：用顺序采样 + batch_size（最简单的情况）
    - shuffle=True ：用自定义的 RandomIndexSampler + BatchSampler 实现打乱
    """

    dataset = MathExprDataset(
        labels_path=labels_path,
        vocab=vocab,
        img_base_dir=Config.IMG_BASE_DIR,
        img_height=Config.IMG_HEIGHT,
        max_width=Config.MAX_WIDTH,
        augment=augment,
    )

    if shuffle:
        # 使用我们自己的随机采样器
        sampler = RandomIndexSampler(dataset)
        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False,   # 最后一个 batch 不足也保留
        )

        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,   # 注意：有了 batch_sampler 就不要再传 batch_size/shuffle
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: math_collate_fn(batch, pad_id=vocab.pad_id),
        )
    else:
        # 不打乱就用最简单的顺序 sampling
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: math_collate_fn(batch, pad_id=vocab.pad_id),
        )

    return loader
