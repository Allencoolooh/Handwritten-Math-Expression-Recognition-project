# train/train.py
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_

from config import Config
from utils.vocab import Vocab
from utils.dataset import create_dataloader
from model.model import MathFormulaRecognizer


def compute_token_accuracy(
        logits: torch.Tensor,  # (B, L, V)
        targets: torch.Tensor,  # (B, L)
        pad_id: int,
) -> float:
    """
    计算 token 级别准确率（忽略 padding）。
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # (B, L)
        mask = (targets != pad_id)  # (B, L)
        correct = (preds == targets) & mask
        total = mask.sum().item()
        if total == 0:
            return 0.0
        return correct.sum().item() / total


def save_checkpoint(
        epoch: int,
        model: MathFormulaRecognizer,
        optimizer: optim.Optimizer,
        best_val_loss: float,
        ckpt_path: Path,
) -> None:
    """
    保存训练 checkpoint。
    """
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ckpt_path)
    print(f"[Checkpoint] Saved to {ckpt_path}")


def load_checkpoint(
        ckpt_path: Path,
        model: MathFormulaRecognizer,
        optimizer: optim.Optimizer,
) -> Dict[str, Any]:
    """
    从 checkpoint 恢复训练（如果需要）。
    返回: {"epoch": 上次结束的 epoch, "best_val_loss": ...}
    """
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", math.inf)
    print(f"[Checkpoint] Loaded from {ckpt_path}, epoch={epoch}, best_val_loss={best_val_loss:.4f}")
    return {"epoch": epoch, "best_val_loss": best_val_loss}


def train_one_epoch(
        model: MathFormulaRecognizer,
        train_loader,
        optimizer: optim.Optimizer,
        epoch: int,
) -> Dict[str, float]:
    """
    训练一个 epoch，返回 {"loss": avg_loss, "acc": avg_acc}。
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0

    start_time = time.time()

    for step, batch in enumerate(train_loader, start=1):
        images = batch["images"].to(Config.DEVICE)  # (B, 1, H, W)
        tgt_input = batch["tgt_input"].to(Config.DEVICE)  # (B, L)
        tgt_output = batch["tgt_output"].to(Config.DEVICE)  # (B, L)
        tgt_lengths = batch["tgt_lengths"].to(Config.DEVICE)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        # 前向
        logits = model(images, tgt_input, tgt_lengths)  # (B, L, V)
        B, L, V = logits.shape

        label_smoothing = getattr(Config, "LABEL_SMOOTHING", 0.1)

        # 计算 loss
        loss = F.cross_entropy(
            logits.view(B * L, V),
            tgt_output.view(B * L),
            ignore_index=model.pad_id,
            label_smoothing=label_smoothing,
        )

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if Config.GRAD_CLIP is not None and Config.GRAD_CLIP > 0:
            clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        # 统计
        with torch.no_grad():
            acc = compute_token_accuracy(logits, tgt_output, pad_id=model.pad_id)
            n_tokens = (tgt_output != model.pad_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_acc += acc * n_tokens
            total_tokens += n_tokens

        if step % 100 == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            avg_acc = total_acc / max(total_tokens, 1)
            elapsed = time.time() - start_time
            print(
                f"[Train] Epoch {epoch} Step {step}/{len(train_loader)} "
                f"Loss {avg_loss:.4f} Acc {avg_acc * 100:.2f}% "
                f"Time {elapsed:.1f}s"
            )

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_acc / max(total_tokens, 1)
    return {"loss": avg_loss, "acc": avg_acc}


def evaluate(
        model: MathFormulaRecognizer,
        val_loader,
) -> Dict[str, float]:
    """
    在验证集上评估，返回 {"loss": avg_loss, "acc": avg_acc}。
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(Config.DEVICE)
            tgt_input = batch["tgt_input"].to(Config.DEVICE)
            tgt_output = batch["tgt_output"].to(Config.DEVICE)
            tgt_lengths = batch["tgt_lengths"].to(Config.DEVICE)

            logits = model(images, tgt_input, tgt_lengths)
            B, L, V = logits.shape

            label_smoothing = getattr(Config, "LABEL_SMOOTHING", 0.1)

            loss = F.cross_entropy(
                logits.view(B * L, V),
                tgt_output.view(B * L),
                ignore_index=model.pad_id,
                label_smoothing=label_smoothing,
            )

            acc = compute_token_accuracy(logits, tgt_output, pad_id=model.pad_id)
            n_tokens = (tgt_output != model.pad_id).sum().item()

            total_loss += loss.item() * n_tokens
            total_acc += acc * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_acc / max(total_tokens, 1)
    return {"loss": avg_loss, "acc": avg_acc}


def main():
    print("Using device:", Config.DEVICE)

    # 1. 加载 vocab
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    print("Vocab size:", len(vocab))

    # 2. 构建 DataLoader
    print("[Data] Building DataLoaders...")
    train_loader = create_dataloader(
        labels_path=Config.TRAIN_LABELS,
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        augment=True,  # ★ 训练集开启数据增强
    )
    val_loader = create_dataloader(
        labels_path=Config.VAL_LABELS,
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        augment=False,  # ★ 验证集不做增强
    )

    # 3. 构建模型 & 优化器
    model = MathFormulaRecognizer(vocab).to(Config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # （可选）从 checkpoint 恢复
    start_epoch = 1
    best_val_loss = math.inf
    resume_path = "../checkpoints/best.pt"  # 如不想恢复训练可改为 None

    if resume_path is not None:
        info = load_checkpoint(Path(resume_path), model, optimizer)
        start_epoch = info["epoch"] + 1
        best_val_loss = info["best_val_loss"]

    # 4. 训练循环
    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{Config.NUM_EPOCHS} =====")

        # 4.1 训练一个 epoch
        train_stats = train_one_epoch(model, train_loader, optimizer, epoch)
        print(
            f"[Train] Epoch {epoch} done. "
            f"Loss {train_stats['loss']:.4f}, Acc {train_stats['acc'] * 100:.2f}%"
        )

        # 4.2 在验证集上评估
        val_stats = evaluate(model, val_loader)
        print(
            f"[Val]   Epoch {epoch} done. "
            f"Loss {val_stats['loss']:.4f}, Acc {val_stats['acc'] * 100:.2f}%"
        )

        # 4.3 保存最新 checkpoint
        ckpt_path = Config.CKPT_DIR / f"last_epoch{epoch:03d}.pt"
        save_checkpoint(epoch, model, optimizer, best_val_loss, ckpt_path)

        # 4.4 如果验证集更好，保存 best checkpoint
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_path = Config.CKPT_DIR / "best.pt"
            save_checkpoint(epoch, model, optimizer, best_val_loss, best_path)
            print(f"[Val] New best model with loss={best_val_loss:.4f}, saved to {best_path}")


if __name__ == "__main__":
    main()

# 12.9模型与训练策略改善后，[Train] Epoch 20 done. Loss 1.0967, Acc 92.68% [Val]   Epoch 20 done. Loss 1.2942, Acc 87.31%

# 12.11 oversampling后，[Train] Epoch 80 done. Loss 1.0596, Acc 93.62%  [Val] Epoch 80 done. Loss 1.1272, Acc 92.37%
