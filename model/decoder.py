# model/decoder.py
from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn

from config import Config


class PositionalEncoding(nn.Module):
    """
    标准的正弦/余弦位置编码（sinusoidal PE），适配 batch_first=True 的张量:
        输入 / 输出形状: (B, L, D)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: (max_len, D)
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )  # (D/2,)

        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # (L, D/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (L, D/2)

        # 为了方便与 batch_first 的 (B, L, D) 相加，存成 (1, L, D)
        pe = pe.unsqueeze(0)  # (1, L, D)
        # register_buffer 表示这是模型参数的一部分，但不参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        # 从 pe 中取前 L 个位置的编码，加到 x 上
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    生成 Decoder 用的“后续位置屏蔽”掩码（下三角为 False，上三角为 True）：
        mask 形状: (L, L)
        mask[i, j] = True  表示在位置 i 计算注意力时不能看到位置 j（j > i）
    """
    # torch.triu 生成上三角，包含对角线；对角线以上为 1，以下为 0
    mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
    return mask  # True 表示被 mask 掉


def create_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    根据每个序列的有效长度 lengths，生成 padding mask。

    输入：
        lengths: (B,) 的 LongTensor，每个元素是该样本的有效长度（不含 padding）
        max_len: 序列的总长度（若为 None，则用 lengths.max()）

    输出：
        mask: (B, max_len) 的 bool 张量
              mask[b, t] = True  表示该位置是 padding，需要在 attention 中被忽略
    """
    if max_len is None:
        max_len = int(lengths.max().item())

    # shape: (B, max_len)，每行是 [0,1,2,...,max_len-1]
    idxs = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(len(lengths), -1)
    # mask: idx >= length → True（padding）
    mask = idxs >= lengths.unsqueeze(1)
    return mask  # (B, max_len), True 为 padding


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器：
    - 输入: 目标 token 序列 (B, L)，编码器输出 memory (B, T_enc, D)
    - 输出: 对每个时间步的 vocab 概率 logits (B, L, vocab_size)

    内部结构：
    - token embedding (含 padding_idx)
    - 正弦位置编码
    - nn.TransformerDecoder (batch_first=True)
    - 线性层映射到 vocab 大小
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.N_HEAD,
        num_layers: int = Config.NUM_DECODER_LAYERS,
        dim_ff: int = Config.DIM_FF,
        dropout: float = Config.DROPOUT,
        pad_id: int = 0,
        max_len: int = 512,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # 1. token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,   # padding token 自动不会参与梯度
        )

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # 3. TransformerDecoder 层（官方实现，batch_first=True）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # 输入输出都是 (B, L, D)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # 4. 输出层：把 decoder 输出的每个时间步特征映射到 vocab 空间
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_input: torch.Tensor,                  # (B, L)，decoder 的输入 token ids（通常是 <SOS> 开头）
        memory: torch.Tensor,                     # (B, T_enc, D)，来自 CNNEncoder+TransformerEncoder
        tgt_key_padding_mask: Optional[torch.Tensor] = None,   # (B, L)，True 表示 padding
        memory_key_padding_mask: Optional[torch.Tensor] = None # (B, T_enc)，True 表示 padding
    ) -> torch.Tensor:
        """
        返回：
            logits: (B, L, vocab_size)
        """
        device = tgt_input.device
        B, L = tgt_input.shape

        # 1. token embedding + 缩放
        #    通常会乘以 sqrt(d_model)，有助于稳定训练
        x = self.embedding(tgt_input) * math.sqrt(self.d_model)   # (B, L, D)

        # 2. 加上位置编码
        x = self.pos_encoder(x)  # (B, L, D)

        # 3. 生成自回归掩码，防止 decoder 在预测第 i 个 token 时看到未来的信息
        #    tgt_mask: (L, L)，True 表示被 mask 掉
        tgt_mask = generate_square_subsequent_mask(L, device=device)

        # 4. 调用官方 TransformerDecoder
        #    注意：batch_first=True，形状都是 (B, L, D)
        decoded = self.decoder(
            tgt=x,                           # (B, L, D)
            memory=memory,                   # (B, T_enc, D)
            tgt_mask=tgt_mask,               # (L, L)
            tgt_key_padding_mask=tgt_key_padding_mask,             # (B, L)
            memory_key_padding_mask=memory_key_padding_mask,       # (B, T_enc)
        )  # -> (B, L, D)

        # 5. 映射到 vocab 空间
        logits = self.output_proj(decoded)  # (B, L, vocab_size)

        return logits
