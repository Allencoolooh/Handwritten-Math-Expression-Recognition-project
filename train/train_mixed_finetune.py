# train/train_mixed_finetune.py
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from config import Config
from utils.vocab import Vocab
from utils.dataset import create_dataloader
from model.model import MathFormulaRecognizer

from loss_utils import build_token_weight_vector, weighted_cross_entropy


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


def train_one_epoch_mixed(
    model: MathFormulaRecognizer,
    loader_a,
    loader_b,
    optimizer: optim.Optimizer,
    device: torch.device,
    weight_vec: torch.Tensor,
) -> dict:
    """
    混合训练一个 epoch：
    - batch 轮流来自 loader_a / loader_b（A,B,A,B,...）
    - 使用加权交叉熵 weighted_cross_entropy（数字/结构/变量/单位加权）
    - 统计 token accuracy（忽略 PAD）
    """
    model.train()

    # 无限迭代器：避免 StopIteration
    def infinite_iter(dl):
        while True:
            for b in dl:
                yield b

    it_a = infinite_iter(loader_a)
    it_b = infinite_iter(loader_b)

    # 每个 epoch 的步数：用两者较小者，避免一边过度主导
    steps = min(len(loader_a), len(loader_b))

    total_loss, total_acc, total_tokens = 0.0, 0.0, 0
    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))
    grad_clip = getattr(Config, "GRAD_CLIP", None)

    start_time = time.time()

    for step in range(steps):
        batch = next(it_a) if (step % 2 == 0) else next(it_b)

        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, tgt_input, tgt_lengths)  # (B,L,V)

        loss = weighted_cross_entropy(
            logits,
            tgt_output,
            pad_id=model.pad_id,
            weight_vec=weight_vec,
            label_smoothing=label_smoothing,
        )

        loss.backward()

        if grad_clip is not None and float(grad_clip) > 0:
            clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        with torch.no_grad():
            acc = token_acc(logits, tgt_output, model.pad_id)
            n_tok = (tgt_output != model.pad_id).sum().item()

            total_loss += loss.item() * n_tok
            total_acc += acc * n_tok
            total_tokens += n_tok

        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"[MixedFT] Step {step}/{steps} "
                f"Loss {total_loss / max(total_tokens,1):.4f} "
                f"Acc {100 * total_acc / max(total_tokens,1):.2f}% "
                f"Time {elapsed:.1f}s"
            )

    return {
        "loss": total_loss / max(total_tokens, 1),
        "acc": total_acc / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate(
    model: MathFormulaRecognizer,
    loader,
    device: torch.device,
    weight_vec: torch.Tensor,
) -> dict:
    """
    验证集评估（与训练保持一致：同样使用加权 CE）。
    返回 {"loss": avg_loss, "acc": avg_acc}
    """
    model.eval()

    total_loss, total_acc, total_tokens = 0.0, 0.0, 0
    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))

    for batch in loader:
        images = batch["images"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        tgt_lengths = batch["tgt_lengths"].to(device)

        logits = model(images, tgt_input, tgt_lengths)  # (B,L,V)

        loss = weighted_cross_entropy(
            logits,
            tgt_output,
            pad_id=model.pad_id,
            weight_vec=weight_vec,
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
    print("[MixedFT] Vocab size:", len(vocab))

    # === DataLoaders ===
    loader_long = create_dataloader(
        labels_path=Path("../data/train_long.txt"),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        augment=True,
    )

    loader_norm = create_dataloader(
        labels_path=Path("../data/train_oversampled.txt"),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        augment=True,
    )

    val_loader = create_dataloader(
        labels_path=Config.VAL_LABELS,
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        augment=False,
    )

    # === Build token weights (critical) ===
    weight_vec = build_token_weight_vector(
        vocab,
        w_digit=1.8,
        w_var=1.3,
        w_struct=1.6,
        w_unit=1.4,
    )

    # === Model ===
    model = MathFormulaRecognizer(vocab).to(device)

    ckpt_path = Path(Config.CKPT_DIR) / "mixedft_best.pt"
    load_model_only(ckpt_path, model, device)

    freeze_encoder(model)
    print("[MixedFT] Encoder frozen, training decoder only.")

    # 只优化 requires_grad=True 的参数（避免冻结导致 optimizer 组不匹配）
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

        tr = train_one_epoch_mixed(
            model, loader_long, loader_norm, optimizer, device, weight_vec
        )
        print(f"[Train] Loss {tr['loss']:.4f} Acc {tr['acc']*100:.2f}%")

        va = evaluate(model, val_loader, device, weight_vec)
        print(f"[Val]   Loss {va['loss']:.4f} Acc {va['acc']*100:.2f}%")

        # 保存 last
        torch.save(
            {"model_state": model.state_dict()},
            ckpt_dir / f"mixedft_last_epoch{ep:03d}.pt",
        )

        # 保存 best（用加权 val loss 判定）
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {"model_state": model.state_dict()},
                ckpt_dir / "mixedft_best.pt",
            )
            print(f"[MixedFT] New best mixed model saved. best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
