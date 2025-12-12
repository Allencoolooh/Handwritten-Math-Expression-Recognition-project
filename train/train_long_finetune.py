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


def get_device():
    if hasattr(Config, "DEVICE"):
        return torch.device(Config.DEVICE)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets.ne(pad_id)
    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def save_checkpoint(
    epoch: int,
    model: MathFormulaRecognizer,
    optimizer: optim.Optimizer,
    best_val_loss: float,
    ckpt_path: Path,
) -> None:
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
    optimizer: optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)

    info = {"epoch": 0, "best_val_loss": math.inf}
    if isinstance(ckpt, dict):
        info["epoch"] = int(ckpt.get("epoch", 0))
        info["best_val_loss"] = float(ckpt.get("best_val_loss", math.inf))

    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state" in ckpt:
        # 注意：我们这次只训练 decoder，参数组可能变了
        # 所以默认不加载 optimizer_state（除非你确认参数组一致）
        pass

    print(f"[Checkpoint] Loaded model weights from {ckpt_path} (epoch={info['epoch']})")
    return info


def freeze_encoder(model: MathFormulaRecognizer) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0
    start = time.time()

    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))

    for step, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, tgt_input, tgt_lengths)  # (B, L, V)
        B, L, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * L, V),
            tgt_output.view(B * L),
            ignore_index=model.pad_id,
            label_smoothing=label_smoothing,
        )

        loss.backward()
        if getattr(Config, "GRAD_CLIP", None):
            clip_grad_norm_(model.parameters(), float(Config.GRAD_CLIP))
        optimizer.step()

        with torch.no_grad():
            acc = compute_token_accuracy(logits, tgt_output, model.pad_id)
            n_tokens = (tgt_output != model.pad_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_acc += acc * n_tokens
            total_tokens += n_tokens

        if step % 50 == 0:
            print(
                f"[TrainLong] Step {step}/{len(loader)} "
                f"Loss {total_loss / max(total_tokens, 1):.4f} "
                f"Acc {100.0 * (total_acc / max(total_tokens, 1)):.2f}% "
                f"Time {time.time() - start:.1f}s"
            )

    return {
        "loss": total_loss / max(total_tokens, 1),
        "acc": total_acc / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_tokens = 0

    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))

    for batch in loader:
        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        logits = model(images, tgt_input, tgt_lengths)
        B, L, V = logits.shape

        loss = F.cross_entropy(
            logits.view(B * L, V),
            tgt_output.view(B * L),
            ignore_index=model.pad_id,
            label_smoothing=label_smoothing,
        )

        acc = compute_token_accuracy(logits, tgt_output, model.pad_id)
        n_tokens = (tgt_output != model.pad_id).sum().item()

        total_loss += loss.item() * n_tokens
        total_acc += acc * n_tokens
        total_tokens += n_tokens

    return {
        "loss": total_loss / max(total_tokens, 1),
        "acc": total_acc / max(total_tokens, 1),
    }


def main():
    device = get_device()
    print("[TrainLong] Using device:", device)

    # 1) vocab
    vocab = Vocab.from_file(Config.VOCAB_PATH)
    print("[TrainLong] Vocab size:", len(vocab))

    # 2) dataloaders
    long_train_path = Path("../data/train_long.txt")
    assert long_train_path.is_file(), (
        "train_long.txt not found. Run:\n"
        "  python tools/build_long_train.py --src data/train_oversampled.txt --dst data/train_long.txt"
    )

    train_loader = create_dataloader(
        labels_path=long_train_path,
        vocab=vocab,
        batch_size=int(getattr(Config, "BATCH_SIZE", 16)),
        shuffle=True,
        num_workers=0,
        augment=True,   # 继续保持增强（但建议强度不要再加太大）
    )

    val_loader = create_dataloader(
        labels_path=Config.VAL_LABELS,
        vocab=vocab,
        batch_size=int(getattr(Config, "BATCH_SIZE", 16)),
        shuffle=False,
        num_workers=0,
        augment=False,
    )

    # 3) model
    model = MathFormulaRecognizer(vocab).to(device)

    # 4) load pretrained weights
    # 这里用你现在 eval_test.py 用的 ckpt：last_epoch080.pt
    ckpt_dir = Path(getattr(Config, "CKPT_DIR", "checkpoints"))
    pretrained_path = ckpt_dir / "last_epoch080.pt"
    assert pretrained_path.is_file(), f"Pretrained ckpt not found: {pretrained_path}"
    load_checkpoint(pretrained_path, model, optimizer=None, device=torch.device("cpu"))

    # 5) freeze encoder
    freeze_encoder(model)
    print("[TrainLong] Encoder frozen. Training decoder only.")

    # 6) optimizer (decoder only)
    lr = float(getattr(Config, "LONG_FT_LR", 5e-5))
    wd = float(getattr(Config, "WEIGHT_DECAY", 0.0))

    decoder_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(decoder_params, lr=lr, weight_decay=wd)

    # 7) train loop
    epochs = int(getattr(Config, "LONG_FT_EPOCHS", 5))
    best_val_loss = math.inf

    out_dir = ckpt_dir
    for epoch in range(1, epochs + 1):
        print(f"\n===== Long Fine-tune Epoch {epoch}/{epochs} =====")

        tr = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[TrainLong] Epoch {epoch} Loss {tr['loss']:.4f} Acc {tr['acc']*100:.2f}%")

        va = evaluate(model, val_loader, device)
        print(f"[Val]      Epoch {epoch} Loss {va['loss']:.4f} Acc {va['acc']*100:.2f}%")

        # save last
        save_checkpoint(epoch, model, optimizer, best_val_loss, out_dir / f"longft_last_epoch{epoch:03d}.pt")

        # save best
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            save_checkpoint(epoch, model, optimizer, best_val_loss, out_dir / "longft_best.pt")
            print(f"[Val] New best long-ft model: loss={best_val_loss:.4f} -> {out_dir / 'longft_best.pt'}")

    print("\n[TrainLong] Done. Next: run eval_test.py after pointing it to longft_best.pt")


if __name__ == "__main__":
    main()
