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


# ----------------------------- #
#          utilities            #
# ----------------------------- #

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
    """
    只加载模型权重，避免 optimizer group mismatch。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    print(f"[MixedFT] Loaded weights from {ckpt_path}")


def freeze_encoder(model: MathFormulaRecognizer):
    for p in model.encoder.parameters():
        p.requires_grad = False


def ss_teacher_prob_by_epoch(epoch: int, start: float, end: float, decay_epochs: int) -> float:
    """
    teacher forcing 概率线性变化：
    epoch=1 -> start
    epoch=decay_epochs -> end
    epoch>decay_epochs -> end
    """
    if decay_epochs <= 1:
        return float(end)
    e = min(max(epoch, 1), decay_epochs)
    t = (e - 1) / float(decay_epochs - 1)  # 0..1
    return float(start + (end - start) * t)


def _dynamic_trim_batch(tgt_input, tgt_output, tgt_lengths, pad_id: int):
    """
    动态裁剪到 batch 内最大有效长度，减少 padding 带来的注意力开销。
    tgt_lengths 表示 tgt_output 的有效长度（不含 PAD）。
    """
    max_valid = int(tgt_lengths.max().item()) if tgt_lengths.numel() > 0 else tgt_input.size(1)
    max_valid = max(2, min(max_valid, tgt_input.size(1)))

    tgt_input = tgt_input[:, :max_valid].contiguous()
    tgt_output = tgt_output[:, :max_valid].contiguous()
    tgt_lengths = torch.clamp(tgt_lengths, min=1, max=max_valid)

    return tgt_input, tgt_output, tgt_lengths


@torch.no_grad()
def _teacher_forcing_pred_ids(
    model: MathFormulaRecognizer,
    images: torch.Tensor,
    tgt_input: torch.Tensor,
    tgt_lengths: torch.Tensor,
    use_amp: bool,
):
    """
    no_grad teacher forcing forward 得到 pred_ids（B,L），用于构造 scheduled input。
    注意：这一步不构建计算图，显存稳定。
    """
    if use_amp and images.is_cuda:
        with torch.cuda.amp.autocast():
            logits_tf = model(images, tgt_input, tgt_lengths)  # (B,L,V)
    else:
        logits_tf = model(images, tgt_input, tgt_lengths)

    pred_ids = logits_tf.argmax(dim=-1)  # (B,L)
    return pred_ids


def _build_mixed_tgt_input(
    tgt_input: torch.Tensor,      # (B,L) teacher forcing 输入（含 SOS）
    pred_ids: torch.Tensor,       # (B,L) teacher forcing 预测 token
    tgt_lengths: torch.Tensor,    # (B,)  tgt_output 有效长度（不含 PAD）
    pad_id: int,
    p_teacher: float,
):
    """
    两段式 Scheduled Sampling：
    - 以 p_teacher 概率使用 GT token，否则替换为 pred token
    - 不替换 SOS（位置 0）
    - 不替换 PAD
    - 不替换超出有效长度的位置
    """
    ss_prob = float(1.0 - p_teacher)
    if ss_prob <= 0.0:
        return tgt_input

    device = tgt_input.device
    B, L = tgt_input.shape

    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B,L)
    alive = pos < tgt_lengths.unsqueeze(1)  # (B,L) 有效位置

    replace = (torch.rand((B, L), device=device) < ss_prob)
    replace[:, 0] = False  # 不替换 SOS
    replace = replace & alive & (tgt_input != pad_id)

    mixed = torch.where(replace, pred_ids, tgt_input)
    return mixed


# ----------------------------- #
#     training / evaluation     #
# ----------------------------- #

def train_one_epoch_mixed(
    model: MathFormulaRecognizer,
    loader_a,
    loader_b,
    optimizer: optim.Optimizer,
    device: torch.device,
    weight_vec: torch.Tensor,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None,
    accum_steps: int,
) -> dict:
    """
    Mixed + Two-pass Scheduled Sampling（稳定版）：
    - A/B 交替 batch
    - no_grad teacher forcing 预测 pred_ids（不进图）
    - 构造 mixed_tgt_input
    - 只对 mixed_tgt_input 做一次有梯度 forward（显存稳定）
    - weighted_cross_entropy + label_smoothing
    - AMP + 梯度累积 + 动态裁剪
    """
    model.train()

    def infinite_iter(dl):
        while True:
            for b in dl:
                yield b

    it_a = infinite_iter(loader_a)
    it_b = infinite_iter(loader_b)
    steps = min(len(loader_a), len(loader_b))

    total_loss, total_acc, total_tokens = 0.0, 0.0, 0
    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))
    grad_clip = getattr(Config, "GRAD_CLIP", None)

    # ===== Scheduled Sampling config =====
    ss_enable = bool(getattr(Config, "SS_ENABLE", True))
    ss_start = float(getattr(Config, "SS_TF_START", 1.0))   # teacher prob start
    ss_end = float(getattr(Config, "SS_TF_END", 0.85))      # 建议先别太低（更稳）
    ss_decay_epochs = int(getattr(Config, "SS_DECAY_EPOCHS", 5))
    p_teacher = ss_teacher_prob_by_epoch(epoch, ss_start, ss_end, ss_decay_epochs) if ss_enable else 1.0

    # AMP
    use_amp = bool(getattr(Config, "AMP_ENABLE", True)) and device.type == "cuda"
    if scaler is None and use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    print(f"[MixedFT] Epoch {epoch}: SS_ENABLE={ss_enable}, teacher_prob={p_teacher:.3f}, AMP={use_amp}, ACCUM={accum_steps}")

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps):
        batch = next(it_a) if (step % 2 == 0) else next(it_b)

        images = batch["images"].to(device, non_blocking=True)
        tgt_input = batch["tgt_input"].to(device, non_blocking=True)
        tgt_output = batch["tgt_output"].to(device, non_blocking=True)
        tgt_lengths = batch["tgt_lengths"].to(device, non_blocking=True)

        # 动态裁剪：减少 padding 注意力计算
        tgt_input, tgt_output, tgt_lengths = _dynamic_trim_batch(
            tgt_input, tgt_output, tgt_lengths, pad_id=model.pad_id
        )

        # ===== Two-pass SS: no_grad TF pred -> mixed input =====
        if ss_enable and p_teacher < 0.999:
            pred_ids = _teacher_forcing_pred_ids(
                model=model,
                images=images,
                tgt_input=tgt_input,
                tgt_lengths=tgt_lengths,
                use_amp=use_amp,
            )
            mixed_tgt_input = _build_mixed_tgt_input(
                tgt_input=tgt_input,
                pred_ids=pred_ids,
                tgt_lengths=tgt_lengths,
                pad_id=model.pad_id,
                p_teacher=p_teacher,
            )
        else:
            mixed_tgt_input = tgt_input

        # ===== One forward with grad =====
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images, mixed_tgt_input, tgt_lengths)
                loss = weighted_cross_entropy(
                    logits,
                    tgt_output,
                    pad_id=model.pad_id,
                    weight_vec=weight_vec,
                    label_smoothing=label_smoothing,
                )
                loss = loss / max(accum_steps, 1)

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                if grad_clip is not None and float(grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), float(grad_clip))

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        else:
            logits = model(images, mixed_tgt_input, tgt_lengths)
            loss = weighted_cross_entropy(
                logits,
                tgt_output,
                pad_id=model.pad_id,
                weight_vec=weight_vec,
                label_smoothing=label_smoothing,
            )
            loss = loss / max(accum_steps, 1)
            loss.backward()

            if (step + 1) % accum_steps == 0:
                if grad_clip is not None and float(grad_clip) > 0:
                    clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # ===== stats =====
        with torch.no_grad():
            loss_scalar = float(loss.item()) * max(accum_steps, 1)
            acc = token_acc(logits, tgt_output, model.pad_id)
            n_tok = (tgt_output != model.pad_id).sum().item()

            total_loss += loss_scalar * n_tok
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

    # 补一次 step（处理 leftover 累积梯度）
    leftover = steps % max(accum_steps, 1)
    if leftover != 0:
        if use_amp and scaler is not None:
            if grad_clip is not None and float(grad_clip) > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            if grad_clip is not None and float(grad_clip) > 0:
                clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss / max(total_tokens, 1),
        "acc": total_acc / max(total_tokens, 1),
        "teacher_prob": p_teacher,
    }


@torch.no_grad()
def evaluate(
    model: MathFormulaRecognizer,
    loader,
    device: torch.device,
    weight_vec: torch.Tensor,
) -> dict:
    """
    验证集：保持 teacher forcing（不做 SS），保证指标可比。
    """
    model.eval()

    total_loss, total_acc, total_tokens = 0.0, 0.0, 0
    label_smoothing = float(getattr(Config, "LABEL_SMOOTHING", 0.1))
    use_amp = bool(getattr(Config, "AMP_ENABLE", True)) and device.type == "cuda"

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        tgt_input = batch["tgt_input"].to(device, non_blocking=True)
        tgt_output = batch["tgt_output"].to(device, non_blocking=True)
        tgt_lengths = batch["tgt_lengths"].to(device, non_blocking=True)

        tgt_input, tgt_output, tgt_lengths = _dynamic_trim_batch(
            tgt_input, tgt_output, tgt_lengths, pad_id=model.pad_id
        )

        if use_amp and images.is_cuda:
            with torch.cuda.amp.autocast():
                logits = model(images, tgt_input, tgt_lengths)
                loss = weighted_cross_entropy(
                    logits,
                    tgt_output,
                    pad_id=model.pad_id,
                    weight_vec=weight_vec,
                    label_smoothing=label_smoothing,
                )
        else:
            logits = model(images, tgt_input, tgt_lengths)
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


# ----------------------------- #
#              main             #
# ----------------------------- #

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
        num_workers=int(getattr(Config, "NUM_WORKERS", 0)),
        augment=True,
    )

    loader_norm = create_dataloader(
        labels_path=Path("../data/train_oversampled.txt"),
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=int(getattr(Config, "NUM_WORKERS", 0)),
        augment=True,
    )

    val_loader = create_dataloader(
        labels_path=Config.VAL_LABELS,
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=int(getattr(Config, "NUM_WORKERS", 0)),
        augment=False,
    )

    # === Token weights ===
    weight_vec = build_token_weight_vector(
        vocab,
        w_digit=float(getattr(Config, "W_DIGIT", 1.8)),
        w_var=float(getattr(Config, "W_VAR", 1.3)),
        w_struct=float(getattr(Config, "W_STRUCT", 1.6)),
        w_unit=float(getattr(Config, "W_UNIT", 1.4)),
    ).to(device)

    # === Model ===
    model = MathFormulaRecognizer(vocab).to(device)

    # 继续训练：优先加载你已有的 best
    ckpt_dir = Path(getattr(Config, "CKPT_DIR", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 允许你从 Config 指定 resume
    resume_name = getattr(Config, "MIXEDFT_RESUME", "mixedft_ss_best.pt")
    ckpt_path = ckpt_dir / str(resume_name)

    if ckpt_path.is_file():
        load_model_only(ckpt_path, model, device)
    else:
        print(f"[MixedFT] Resume not found: {ckpt_path}. Start from random init.")

    # 冻结 encoder（只训 decoder）
    if bool(getattr(Config, "FREEZE_ENCODER", True)):
        freeze_encoder(model)
        print("[MixedFT] Encoder frozen, training decoder only.")
    else:
        print("[MixedFT] Encoder NOT frozen (training full model).")

    # optimizer：只优化 requires_grad=True
    lr = float(getattr(Config, "MIXEDFT_LR", 3e-5))
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(getattr(Config, "WEIGHT_DECAY", 0.0)),
    )

    # AMP + GradScaler
    use_amp = bool(getattr(Config, "AMP_ENABLE", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 梯度累积（减少 OOM）
    accum_steps = int(getattr(Config, "ACCUM_STEPS", 2 if use_amp else 1))
    accum_steps = max(1, accum_steps)

    # epochs
    epochs = int(getattr(Config, "MIXEDFT_EPOCHS", 5))
    best_val = math.inf

    print(f"[MixedFT] LR={lr} AMP={use_amp} ACCUM_STEPS={accum_steps} EPOCHS={epochs}")

    for ep in range(1, epochs + 1):
        print(f"\n===== Mixed Fine-tune (SS Two-pass) Epoch {ep}/{epochs} =====")

        tr = train_one_epoch_mixed(
            model=model,
            loader_a=loader_long,
            loader_b=loader_norm,
            optimizer=optimizer,
            device=device,
            weight_vec=weight_vec,
            epoch=ep,
            scaler=scaler,
            accum_steps=accum_steps,
        )
        print(f"[Train] Loss {tr['loss']:.4f} Acc {tr['acc']*100:.2f}% teacher_prob={tr['teacher_prob']:.3f}")

        va = evaluate(model, val_loader, device, weight_vec)
        print(f"[Val]   Loss {va['loss']:.4f} Acc {va['acc']*100:.2f}%")

        # save last
        torch.save(
            {"model_state": model.state_dict()},
            ckpt_dir / f"mixedft_ss_last_epoch{ep:03d}.pt",
        )

        # save best
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {"model_state": model.state_dict()},
                ckpt_dir / "mixedft_ss_best.pt",
            )
            print(f"[MixedFT] New best saved. best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
