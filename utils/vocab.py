# utils/vocab.py
from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

from config import Config


class Vocab:
    """
    负责：LaTeX 文本 <-> token 序列 <-> id 序列。

    使用示例：
        vocab = Vocab.build_from_labels_file(Config.FULL_LABELS, save_path=Config.VOCAB_PATH)
        vocab = Vocab.from_file(Config.VOCAB_PATH)
        ids = vocab.encode(r"\\frac{1}{x^2}")
    """

    def __init__(
        self,
        token2id: Dict[str, int],
        pad_token: str = Config.PAD_TOKEN,
        sos_token: str = Config.SOS_TOKEN,
        eos_token: str = Config.EOS_TOKEN,
        unk_token: str = Config.UNK_TOKEN,
    ):
        self.token2id = token2id
        self.id2token = [None] * len(token2id)
        for tok, idx in token2id.items():
            self.id2token[idx] = tok

        # 特殊符号
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.pad_id = token2id[pad_token]
        self.sos_id = token2id[sos_token]
        self.eos_id = token2id[eos_token]
        self.unk_id = token2id[unk_token]

    # ----------------------------------------------------------------------
    #                          LaTeX Tokenizer
    # ----------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        LaTeX 分词，特点：

        1）彻底过滤掉非 ASCII 字符（包括中文、全角符号）。
        2）识别 LaTeX 命令（\frac, \sqrt, \sin, \alpha 等）作为一个 token。
        3）特殊符号 {}[]()^_+-=*/=<>.,;:!|% 单字符 token。
        4）数字逐字符 token。
        5）普通字母逐字符 token。
        """

        # 1) 过滤所有非 ASCII 字符
        text = "".join(ch for ch in text if ch.isascii())

        tokens: List[str] = []
        i = 0
        L = len(text)

        SPECIAL_CHARS = set("{}[]()^_+-=*/=<>.,;:!|%")

        while i < L:
            ch = text[i]

            # 跳过空白
            if ch.isspace():
                i += 1
                continue

            # 2) LaTeX 命令：以反斜杠开头 + 连续字母
            if ch == "\\":
                j = i + 1
                while j < L and text[j].isalpha():
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue

            # 3) 特殊单字符 token
            if ch in SPECIAL_CHARS:
                tokens.append(ch)
                i += 1
                continue

            # 4) 数字逐个 token
            if ch.isdigit():
                tokens.append(ch)
                i += 1
                continue

            # 5) 字母逐个 token
            if ch.isalpha():
                tokens.append(ch)
                i += 1
                continue

            # 6) 其它符号（?, ~ 等），作为单字符 token
            tokens.append(ch)
            i += 1

        return tokens

    # ----------------------------------------------------------------------
    #                 构建 Vocab（从标注文件统计 token）
    # ----------------------------------------------------------------------
    @classmethod
    def build_from_labels_file(
        cls,
        labels_path: Path,
        min_freq: int = 1,
        save_path: Optional[Path] = None,
    ) -> "Vocab":
        labels_path = Path(labels_path)
        assert labels_path.is_file(), f"Labels file not found: {labels_path}"

        counter = Counter()

        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    _, latex = line.split("\t", 1)
                except ValueError:
                    continue

                latex = latex.strip()
                tokens = cls._tokenize(latex)
                counter.update(tokens)

        # 过滤低频 & 只保留 ASCII token（双保险）
        candidates = [tok for tok, freq in counter.items() if freq >= min_freq]
        candidates = [tok for tok in candidates if all(ch.isascii() for ch in tok)]

        # 排序：先按出现次数，从大到小；再按字典序
        sorted_tokens = sorted(candidates, key=lambda t: (-counter[t], t))

        # 先加入特殊符号
        special_tokens = [
            Config.PAD_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
            Config.UNK_TOKEN,
        ]

        vocab_tokens = []
        seen = set()

        for tok in special_tokens + sorted_tokens:
            if tok not in seen:
                vocab_tokens.append(tok)
                seen.add(tok)

        token2id = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        vocab = cls(token2id)

        if save_path is not None:
            vocab.save(save_path)

        return vocab

    # ----------------------------------------------------------------------
    #                            加载 Vocab
    # ----------------------------------------------------------------------
    @classmethod
    def from_file(cls, vocab_path: Path) -> "Vocab":
        vocab_path = Path(vocab_path)
        assert vocab_path.is_file(), f"Vocab file not found: {vocab_path}"

        tokens = []
        with vocab_path.open("r", encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n")
                if tok:
                    tokens.append(tok)

        token2id = {tok: idx for idx, tok in enumerate(tokens)}
        return cls(token2id)

    # ----------------------------------------------------------------------
    #                           保存 vocab.txt
    # ----------------------------------------------------------------------
    def save(self, save_path: Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w", encoding="utf-8") as f:
            for tok in self.id2token:
                f.write(tok + "\n")

        print(f"[Vocab] saved to {save_path} (size={len(self)})")

    # ----------------------------------------------------------------------
    #                        文本编码 / 解码
    # ----------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]

        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            if idx < 0 or idx >= len(self.id2token):
                continue
            tok = self.id2token[idx]
            if remove_special_tokens and tok in {
                self.pad_token,
                self.sos_token,
                self.eos_token,
            }:
                continue
            tokens.append(tok)

        # token 级别直接拼接
        return "".join(tokens)

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.id2token)


# --------------------------------------------------------------------------
#                     CLI：命令行自动构建 vocab
# --------------------------------------------------------------------------
if __name__ == "__main__":
    vocab = Vocab.build_from_labels_file(
        labels_path=Config.FULL_LABELS,
        min_freq=1,
        save_path=Config.VOCAB_PATH,
    )
    print("Vocab size =", len(vocab))
