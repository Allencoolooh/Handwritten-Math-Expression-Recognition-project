# train/train_mixed_finetune.py
from __future__ import annotations

import math
import time
from pathlib import Path

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
def token_acc(logits, targets, pad_id):
    preds = logits.argmax(dim=-1)
    mask = targets.ne(pad_id)
    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def load_model_only(ckpt_path: Path, model: MathFormulaRecognizer, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    print(f"[MixedFT] Loaded weights from {ckpt_path}")


def freeze_encoder(model: MathFormulaRecognizer):
    for p in model.encoder.parameters():
        p.requires_grad = False


def train_one_epoch_mixed(model, loader_a, loader_b, optimizer, device):
    model.train()
    it_a = iter(loader_a)
    it_b = iter(loader_b)

    steps = min(len(loader_a), len(loader_b))
    total_loss, total_acc, total_tokens = 0.0, 0.0, 0

    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))

    for step in range(steps):
        # === 轮流取 batch：A -> B ===
        batch = next(it_a) if step % 2 == 0 else next(it_b)

        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, tgt_input, tgt_lengths)
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
            acc = token_acc(logits, tgt_output, model.pad_id)
            n_tok = (tgt_output != model.pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_acc += acc * n_tok
            total_tokens += n_tok

        if step % 50 == 0:
            print(
                f"[MixedFT] Step {step}/{steps} "
                f"Loss {total_loss / max(total_tokens,1):.4f} "
                f"Acc {100 * total_acc / max(total_tokens,1):.2f}%"
            )

    return {
        "loss": total_loss / max(total_tokens, 1),
        "acc": total_acc / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, total_tokens = 0.0, 0.0, 0

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

        acc = token_acc(logits, tgt_output, model.pad_id)
        n_tok = (tgt_output != model.pad_id).sum().item()

        total_loss += loss.item() * n_tok
        total_acc += acc * n_tok
        total_tokens += n_tok

    return {
        "loss": total_loss / max(total_tokens, 1),
        "acc": total_acc / max(total_tokens, 1),
    }


def main():
    device = get_device()
    print("[MixedFT] Using device:", device)

    vocab = Vocab.from_file(Config.VOCAB_PATH)

    # === DataLoaders ===
    loader_long = create_dataloader(
        labels_path=Path("../data/train_long.txt"),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        augment=True,
    )

    loader_norm = create_dataloader(
        labels_path=Path("../data/train_oversampled.txt"),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        augment=True,
    )

    val_loader = create_dataloader(
        labels_path=Config.VAL_LABELS,
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        augment=False,
    )

    # === Model ===
    model = MathFormulaRecognizer(vocab).to(device)

    ckpt_path = Path(Config.CKPT_DIR) / "longft_last_epoch005.pt"
    load_model_only(ckpt_path, model, device)

    freeze_encoder(model)
    print("[MixedFT] Encoder frozen, training decoder only.")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=getattr(Config, "WEIGHT_DECAY", 0.0),
    )

    epochs = 5
    best_val = math.inf
    ckpt_dir = Path(Config.CKPT_DIR)

    for ep in range(1, epochs + 1):
        print(f"\n===== Mixed Fine-tune Epoch {ep}/{epochs} =====")

        tr = train_one_epoch_mixed(model, loader_long, loader_norm, optimizer, device)
        print(f"[Train] Loss {tr['loss']:.4f} Acc {tr['acc']*100:.2f}%")

        va = evaluate(model, val_loader, device)
        print(f"[Val]   Loss {va['loss']:.4f} Acc {va['acc']*100:.2f}%")

        torch.save(
            {"model_state": model.state_dict()},
            ckpt_dir / f"mixedft_last_epoch{ep:03d}.pt"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {"model_state": model.state_dict()},
                ckpt_dir / "mixedft_best.pt"
            )
            print("[MixedFT] New best mixed model saved.")


if __name__ == "__main__":
    main()
