# model/model.py
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .encoder import CNNEncoder
from .decoder import TransformerDecoder, create_padding_mask


class MathFormulaRecognizer(nn.Module):
    """
    整体模型：
    - 图像输入: images  (B, 1, H, W)
    - 文本输入: tgt_input (B, L)  —— 训练时, 形如 [SOS, token1, ..., token_{L-1}]
    - 图像编码: CNNEncoder -> memory (B, T_enc, D)
    - 文本解码: TransformerDecoder(memory, tgt_input) -> logits (B, L, vocab_size)

    使用场景：
    - 训练 forward(): 传入 images, tgt_input, tgt_lengths，拿到 logits 用 CE 计算 loss。
    - 推理 greedy_decode(): 只给 images，自动从 <SOS> 开始一步步生成 LaTeX。
    """

    def __init__(
        self,
        vocab,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.N_HEAD,
        num_decoder_layers: int = Config.NUM_DECODER_LAYERS,
        dim_ff: int = Config.DIM_FF,
        dropout: float = Config.DROPOUT,
    ) -> None:
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_id = vocab.pad_id
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id

        # 图像编码器
        self.encoder = CNNEncoder(
            in_channels=Config.NUM_CHANNELS,
            d_model=d_model,
        )

        # 文本解码器
        self.decoder = TransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            pad_id=self.pad_id,
            max_len=Config.MAX_TGT_LEN,   # 你可以在 Config 里定义 MAX_TGT_LEN
        )

    # ------------------------------------------------------------------ #
    #                           训练前向                                 #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,          # (B, 1, H, W)
        tgt_input: torch.Tensor,       # (B, L)  —— decoder 输入序列
        tgt_lengths: torch.Tensor,     # (B,)    —— 对应 tgt_output 的有效长度（不含 padding）
    ) -> torch.Tensor:
        """
        返回:
            logits: (B, L, vocab_size)
        之后你可以用 CrossEntropyLoss(ignore_index=pad_id) 来算 loss：
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                tgt_output.view(-1),
                ignore_index=pad_id,
            )
        """
        # 1. 图像 -> 序列特征 (B, T_enc, D)
        memory = self.encoder(images)  # (B, T_enc, D)

        # 2. decoder 侧 padding mask（针对 tgt_input / tgt_output）
        #    tgt_lengths 对应的是 y_out 的有效长度（我们在 collate_fn 里已经减 1 了）
        #    这里的 max_len = L = tgt_input.size(1)
        tgt_key_padding_mask = create_padding_mask(
            lengths=tgt_lengths,
            max_len=tgt_input.size(1),
        )  # (B, L)，True 表示 padding

        # 3. 调用 TransformerDecoder
        logits = self.decoder(
            tgt_input=tgt_input,                      # (B, L)
            memory=memory,                            # (B, T_enc, D)
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,             # encoder 侧目前无需 padding mask
        )  # (B, L, vocab_size)

        return logits

    # ------------------------------------------------------------------ #
    #                         贪心解码 (推理)                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,             # (B, 1, H, W)
        max_len: int = 128,
        device: Optional[torch.device] = None,
    ) -> List[List[int]]:
        """
        给定图像，使用贪心策略一步步生成 token 序列（只返回 id 序列）。

        过程：
        - 第一步输入 [SOS]，解码出第一个 token
        - 每次取 argmax 作为下一个 token，拼到序列末尾
        - 若某个样本生成了 EOS，则后续步骤不再更新它（保持 EOS 之后的 token 也为 EOS）
        """
        if device is None:
            device = images.device

        self.eval()

        B = images.size(0)
        images = images.to(device)

        # 1. 编码图像
        memory = self.encoder(images)  # (B, T_enc, D)

        # 2. 初始输入：全是 SOS
        ys = torch.full(
            (B, 1), fill_value=self.sos_id, dtype=torch.long, device=device
        )  # (B, 1)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            L = ys.size(1)

            # 当前长度都相同，tgt_lengths = L for all
            tgt_lengths = torch.full(
                (B,), fill_value=L, dtype=torch.long, device=device
            )

            # 生成 padding mask
            tgt_key_padding_mask = create_padding_mask(
                lengths=tgt_lengths,
                max_len=L,
            )  # (B, L)

            # 解码得到 logits: (B, L, vocab_size)
            logits = self.decoder(
                tgt_input=ys,
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None,
            )

            # 取最后一个时间步的概率
            last_logits = logits[:, -1, :]           # (B, vocab_size)
            next_tokens = last_logits.argmax(dim=-1) # (B,)

            # 对已经 finished 的样本，强制保持 EOS，不再改变
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.eos_id),
                next_tokens,
            )

            # 拼接到序列末尾
            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)  # (B, L+1)

            # 更新 finished 标记：如果本步产生了 EOS，则记为完成
            finished = finished | (next_tokens == self.eos_id)

            # 如果所有样本都已经生成 EOS，可以提前结束
            if finished.all():
                break

        # ys 形如 [SOS, token1, ..., tokenK, (可能有 EOS ...)]
        # 返回 Python list 形式
        return [seq.tolist() for seq in ys]

    @torch.no_grad()
    def recognize(
        self,
        images: torch.Tensor,
        max_len: int = 128,
        device: Optional[torch.device] = None,
    ) -> List[str]:
        """
        在 greedy_decode 的基础上，直接返回解码后的 LaTeX 字符串列表。
        """
        seq_ids_batch = self.greedy_decode(images, max_len=max_len, device=device)

        texts: List[str] = []
        for ids in seq_ids_batch:
            # decode 会默认去掉 <SOS>/<EOS>/<PAD>
            text = self.vocab.decode(ids, remove_special_tokens=True)
            texts.append(text)
        return texts
