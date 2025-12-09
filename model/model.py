# model/model.py
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .encoder import CNNEncoder, ResNetEncoder
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
        self.encoder = ResNetEncoder(
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
            max_len=Config.MAX_TGT_LEN,  # 你可以在 Config 里定义 MAX_TGT_LEN
        )

    # ------------------------------------------------------------------ #
    #                           训练前向                                 #
    # ------------------------------------------------------------------ #
    def forward(
            self,
            images: torch.Tensor,  # (B, 1, H, W)
            tgt_input: torch.Tensor,  # (B, L)  —— decoder 输入序列
            tgt_lengths: torch.Tensor,  # (B,)    —— 对应 tgt_output 的有效长度（不含 padding）
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
            tgt_input=tgt_input,  # (B, L)
            memory=memory,  # (B, T_enc, D)
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,  # encoder 侧目前无需 padding mask
        )  # (B, L, vocab_size)

        return logits

    # ------------------------------------------------------------------ #
    #                         贪心解码 (推理)                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_decode(
            self,
            images: torch.Tensor,  # (B, 1, H, W)
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
            last_logits = logits[:, -1, :]  # (B, vocab_size)
            next_tokens = last_logits.argmax(dim=-1)  # (B,)

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

    # ------------------------------------------------------------------ #
    #                        Beam Search 解码                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def beam_search_decode(
            self,
            images: torch.Tensor,  # (B, 1, H, W)
            max_len: int = 128,
            device: Optional[torch.device] = None,
            beam_size: int = 3,
    ) -> List[List[int]]:
        """
        使用 Beam Search 进行解码，返回每张图的 token id 序列。
        优化版：对每个样本，每个 step 只调用一次 decoder，
        将该样本的所有 beam 序列一起打包成一个 mini-batch。
        """
        if device is None:
            device = images.device

        self.eval()
        images = images.to(device)

        # 1. 图像编码
        memory_all = self.encoder(images)  # (B, T_enc, D)
        B = memory_all.size(0)

        results: List[List[int]] = []

        for b in range(B):
            # 取出单张图片对应的 encoder memory
            memory = memory_all[b:b + 1]  # (1, T_enc, D)

            # beam: list of (score, seq_ids)
            # 初始只有一个 beam，内容为 [SOS]
            beam = [(0.0, [self.sos_id])]

            for _ in range(max_len):
                # 如果当前所有 beam 都已经以 EOS 结尾，直接结束
                if all(seq[-1] == self.eos_id for _, seq in beam):
                    break

                # 1) 把所有 beam 序列打包成一个 batch
                beam_len_list = [len(seq) for _, seq in beam]
                max_beam_len = max(beam_len_list)
                K = len(beam)  # 当前 beam 数（首次为 1，之后一般为 beam_size）

                tgt_input = torch.full(
                    (K, max_beam_len),
                    fill_value=self.pad_id,
                    dtype=torch.long,
                    device=device,
                )  # (K, L_max)

                for i, (_, seq) in enumerate(beam):
                    L_i = len(seq)
                    tgt_input[i, :L_i] = torch.tensor(seq, dtype=torch.long, device=device)

                tgt_lengths = torch.tensor(beam_len_list, dtype=torch.long, device=device)

                # 2) 为这 K 条序列构造 padding mask
                tgt_key_padding_mask = create_padding_mask(
                    lengths=tgt_lengths,
                    max_len=max_beam_len,
                )  # (K, L_max)

                # 3) 扩展 memory 到 K 条（同一张图，每个 beam 共享同一个 encoder memory）
                memory_expand = memory.expand(K, -1, -1)  # (K, T_enc, D)

                # 4) 一次性跑 decoder，拿到所有 beam 在当前 step 的 logits
                logits = self.decoder(
                    tgt_input=tgt_input,
                    memory=memory_expand,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=None,
                )  # (K, L_max, V)

                # 5) 取每条 beam 当前序列最后一个非 pad 位置的 logits
                last_logits_list = []
                for i, L_i in enumerate(beam_len_list):
                    last_logits_list.append(logits[i, L_i - 1, :].unsqueeze(0))  # (1, V)
                last_logits = torch.cat(last_logits_list, dim=0)  # (K, V)

                log_probs = F.log_softmax(last_logits, dim=-1)  # (K, V)

                # 6) 对每条 beam 分别做 top-k 扩展
                new_beam = []
                for i, (score, seq) in enumerate(beam):
                    # 若该 beam 已经结束，则直接保留
                    if seq[-1] == self.eos_id:
                        new_beam.append((score, seq))
                        continue

                    lp_i, idx_i = torch.topk(log_probs[i], beam_size)  # (beam_size,)
                    for lp, idx in zip(lp_i.tolist(), idx_i.tolist()):
                        new_score = score + lp
                        new_seq = seq + [idx]
                        new_beam.append((new_score, new_seq))

                # 7) 从 new_beam 中选出得分最高的 beam_size 条
                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:beam_size]

            # 选出分数最高的一条作为最终结果
            best_score, best_seq = max(beam, key=lambda x: x[0])

            # 去掉开头的 SOS，末尾若有 EOS 也去掉
            if best_seq and best_seq[0] == self.sos_id:
                best_seq = best_seq[1:]
            if best_seq and best_seq[-1] == self.eos_id:
                best_seq = best_seq[:-1]

            results.append(best_seq)

        return results

    @torch.no_grad()
    def recognize_beam(
            self,
            images: torch.Tensor,
            max_len: int = 128,
            device: Optional[torch.device] = None,
            beam_size: int = 3,
    ) -> List[str]:
        """
        基于 Beam Search 的高质量解码接口，直接返回 LaTeX 字符串列表。
        """
        seq_ids_batch = self.beam_search_decode(
            images,
            max_len=max_len,
            device=device,
            beam_size=beam_size,
        )

        texts: List[str] = []
        for ids in seq_ids_batch:
            text = self.vocab.decode(ids, remove_special_tokens=True)
            texts.append(text)
        return texts
