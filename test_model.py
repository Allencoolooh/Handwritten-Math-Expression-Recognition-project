import torch
import torch.nn.functional as F

from config import Config
from utils.vocab import Vocab
from utils.dataset import create_dataloader
from model.model import MathFormulaRecognizer


def main():
    # 1. 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. 加载词表
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    print("Vocab size:", len(vocab))

    # 3. 构建一个小的 DataLoader（batch_size=2，方便看）
    train_loader = create_dataloader(
        labels_path=Config.TRAIN_LABELS,
        vocab=vocab,
        batch_size=2,
        shuffle=True,      # 测试一下有 shuffle 的版本
        num_workers=0,
        augment=False,
    )

    # 取一个 batch
    batch = next(iter(train_loader))
    images      = batch["images"].to(device)       # (B, 1, H, W)
    tgt_input   = batch["tgt_input"].to(device)    # (B, L-1)
    tgt_output  = batch["tgt_output"].to(device)   # (B, L-1)
    tgt_lengths = batch["tgt_lengths"].to(device)  # (B,)
    labels      = batch["labels"]                  # 原始 LaTeX 文本 list[str]

    print("images shape    :", images.shape)
    print("tgt_input shape :", tgt_input.shape)
    print("tgt_output shape:", tgt_output.shape)
    print("tgt_lengths     :", tgt_lengths)
    print("labels          :", labels)

    # 4. 构建模型
    model = MathFormulaRecognizer(vocab).to(device)
    model.train()   # 测 `forward + loss` 的逻辑，设为 train 模式即可

    # 5. 前向传播
    logits = model(images, tgt_input, tgt_lengths)    # (B, L-1, vocab_size)
    B, L, V = logits.shape
    print("logits shape    :", logits.shape)

    # 6. 计算一个交叉熵损失，检查维度是否匹配
    loss = F.cross_entropy(
        logits.view(B * L, V),
        tgt_output.view(B * L),
        ignore_index=vocab.pad_id,
    )
    print("Dummy loss      :", loss.item())

    # 7. 测试一次贪心解码（推理），看能否完整跑通
    model.eval()
    with torch.no_grad():
        # 用同一个 batch 的图像做推理
        preds = model.recognize(images, max_len=Config.MAX_TGT_LEN, device=device)
        print("\n=== Greedy decode results ===")
        for i, (gt, pr) in enumerate(zip(labels, preds)):
            print(f"[{i}] GT :", gt)
            print(f"    PR :", pr)


if __name__ == "__main__":
    main()
