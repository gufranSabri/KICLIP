from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.config import Config
from src.data.dataset import VideoDataset
from src.data.prompts import TEMPLATES, tokenize
from src.models.kiclip import KICLIP, build_kiclip
from src.test import evaluate
from src.utils import dist
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import format_metrics
from src.utils.progress import progress
from src.utils.seed import seed_everything
from src.utils.snapshot import snapshot_code

logger = get_logger(__name__)


def lr_at(cfg: Config, epoch: float) -> float:
    t = cfg.train
    if epoch < t.warmup_epochs:
        return t.warmup_start_lr + (t.lr - t.warmup_start_lr) * epoch / t.warmup_epochs
    frac = (epoch - t.warmup_epochs) / (t.epochs - t.warmup_epochs)
    return t.end_lr + (t.lr - t.end_lr) * (1 + math.cos(math.pi * frac)) / 2


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    # biases and 1-D params skip weight decay; logit_scale is 0-D and keeps it, as in FROSTER
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _accumulate(average: dict | None, model: torch.nn.Module, n: int) -> dict:
    state = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
    if average is None:
        return state
    for k, v in state.items():
        average[k] += (v - average[k]) / n
    return average


def train_epoch(model, loader, tokens, optimizer, scaler, cfg: Config, epoch: int, accum: int,
                device: torch.device, main: bool) -> dict:
    model.train()
    amp = cfg.train.amp and device.type == "cuda"
    ddp = isinstance(model, DistributedDataParallel)
    steps = len(loader) // accum
    batches = iter(loader)
    sums = torch.zeros(5, device=device)  # loss, ce, kd, correct, videos

    for step in progress(range(steps), desc=f"epoch {epoch + 1}", leave=False, ncols=100, disable=not main):
        lr = lr_at(cfg, epoch + step / steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)

        for micro in range(accum):
            video, labels = next(batches)
            video, labels = video.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # one random template plus the bare description, redrawn every forward
            prompt = tokens[[random.randrange(len(TEMPLATES)), len(TEMPLATES)]]
            sync = contextlib.nullcontext() if not ddp or micro == accum - 1 else model.no_sync()
            with sync:
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    logits, kd = model(video, prompt)
                    ce = F.cross_entropy(logits, labels)
                    loss = ce + cfg.model.distill_weight * kd
                scaler.scale(loss / accum).backward()
            step_loss += loss.detach()
            sums += torch.stack([
                loss.detach(), ce.detach(), kd.detach(),
                (logits.argmax(-1) == labels).sum().float(), torch.tensor(float(len(labels)), device=device),
            ])

        if not torch.isfinite(step_loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch + 1}, step {step}; lower train.lr or disable train.amp")
        scaler.unscale_(optimizer)
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    loss, ce, kd, correct, videos = dist.all_sum(sums.tolist(), device)
    micro_steps = steps * accum * dist.rank_world()[1]
    return {
        "loss": loss / micro_steps, "ce": ce / micro_steps, "kd": kd / micro_steps,
        "top1": 100 * correct / max(videos, 1), "lr": lr,
    }


def _section(title: str) -> None:
    logger.info("")
    logger.info("--- %s %s", title, "-" * max(0, 58 - len(title)))


def log_header(cfg: Config, run_dir, device, world: int, accum: int, steps: int, train_set, val_set) -> None:
    m, d, t = cfg.model, cfg.data, cfg.train
    _section("run")
    logger.info("run dir      : %s", run_dir)
    logger.info("device       : %s x %d", device, world)
    logger.info("seed         : %d", cfg.seed)

    _section("data")
    logger.info("train        : %s (%d videos, %d classes) from %s",
                d.train, len(train_set), len(train_set.classnames), train_set.root)
    logger.info("val          : %s", f"{d.val} every {t.eval_every} epoch(s)" if val_set is not None else "off")
    logger.info("clip         : %d frames x %d @ %g fps = %.2f s | crop %d | scales %s | flip %s",
                d.num_frames, d.sampling_rate, d.target_fps, train_set.span, d.crop_size, d.train_scales, d.flip)

    _section("model")
    logger.info("student      : %s OpenAI CLIP | temporal %s | grad checkpointing %s",
                m.arch, m.temporal, m.grad_checkpointing)
    source = {"teacher": "the teacher's activations", "blocks": "frozen block copies run on the student's stem"}
    logger.info("injection    : %s", f"{source[m.inject_source]}, averaged in before every vision block"
                if m.inject else "off")
    logger.info("teacher      : %s", f"frozen {m.arch} | distill_weight {m.distill_weight:g} | "
                f"residual_alpha {m.residual_alpha:g}"
                if m.distill_weight > 0 or (m.inject and m.inject_source == "teacher") else "off")

    _section("optimisation")
    logger.info("batch        : %d = %d per forward x %d GPU(s) x %d accumulation",
                t.batch_size, t.micro_batch_size, world, accum)
    logger.info("steps        : %d per epoch x %d epochs", steps, t.epochs)
    logger.info("lr           : %.3g | warmup %g epochs from %.3g | cosine to %.3g",
                t.lr, t.warmup_epochs, t.warmup_start_lr, t.end_lr)
    logger.info("regularise   : weight_decay %g | grad_clip %g | amp %s", t.weight_decay, t.grad_clip, t.amp)
    logger.info("average      : epochs %d..%d -> averaged.pt", t.average_from, t.epochs)


def log_param_counts(model: KICLIP) -> None:
    clip = model.model
    groups = [
        ("vision encoder", [clip.visual]),
        ("text encoder", [clip.transformer, clip.token_embedding, clip.ln_final]),
        ("projectors", [p for p in (model.projector_v, model.projector_t) if p is not None]),
    ]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, mods in groups:
        n = sum(p.numel() for mod in mods for p in mod.parameters() if p.requires_grad)
        logger.info("  %-22s %12s  (%4.1f%%)", name, f"{n:,}", 100 * n / max(total, 1))
    logger.info("  %-22s %12s", "TOTAL trainable", f"{total:,}")
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("  %-22s %12s  (teacher, injection blocks)", "frozen", f"{frozen:,}")


def train(cfg: Config, run_name: str | None = None) -> dict:
    rank, world, device = dist.setup()
    main = rank == 0
    seed_everything(cfg.seed + rank)
    t = cfg.train

    if t.batch_size % (t.micro_batch_size * world):
        raise ValueError(
            f"train.batch_size ({t.batch_size}) must be a multiple of train.micro_batch_size x GPUs "
            f"({t.micro_batch_size} x {world})"
        )
    accum = t.batch_size // (t.micro_batch_size * world)

    run_dir = cfg.run_dir(run_name)
    if main:
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot_code(run_dir)
        setup_logging(log_file=run_dir / "train.log")
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    else:
        logging.getLogger().setLevel(logging.WARNING)
    dist.barrier()

    # ---- data ---------------------------------------------------------
    train_set = VideoDataset(cfg, cfg.data.train, train=True)
    tokens = tokenize(train_set.classnames).to(device)  # (prompts, C, L)
    sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.seed, drop_last=True) if world > 1 else None
    loader = DataLoader(
        train_set,
        batch_size=t.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.data.num_workers > 0,
    )
    steps = len(loader) // accum
    if steps == 0:
        raise ValueError(f"{cfg.data.train} has {len(train_set)} videos, fewer than one batch of {t.batch_size}")
    val_set = VideoDataset(cfg, cfg.data.val, train=False, views=1) if t.eval_every else None
    val_tokens = tokenize(val_set.classnames) if val_set is not None else None

    if main:
        log_header(cfg, run_dir, device, world, accum, steps, train_set, val_set)

    # ---- model --------------------------------------------------------
    student = build_kiclip(cfg).to(device)
    if world > 1:
        model = DistributedDataParallel(student, device_ids=[device.index] if device.type == "cuda" else None)
    else:
        model = student
    optimizer = torch.optim.AdamW(param_groups(student, t.weight_decay), lr=t.lr, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler(device.type, enabled=t.amp and device.type == "cuda")

    if main:
        _section("parameters")
        log_param_counts(student)

    last = run_dir / "last.pt"
    start, history, average, averaged = 0, [], None, 0
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["config"] != cfg.to_dict():
            raise ValueError(f"{last} was written with a different config; delete it or pass another --run-name")
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start, history, averaged = ckpt["epoch"], ckpt["history"], ckpt["averaged"]
        average = ckpt["average"] if main else None
        logger.info("resuming from %s after epoch %d", last, start)
        del ckpt

    # ---- loop ---------------------------------------------------------
    _section("training")
    for epoch in range(start, t.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        record = {"epoch": epoch + 1, **train_epoch(
            model, loader, tokens, optimizer, scaler, cfg, epoch, accum, device, main
        )}
        if main and epoch + 1 >= t.average_from:
            averaged += 1
            average = _accumulate(average, student, averaged)
        if val_set is not None and (epoch + 1) % t.eval_every == 0:
            record["val"] = evaluate(student, val_set, val_tokens, cfg, device, desc=f"epoch {epoch + 1} val")
        record["seconds"] = time.time() - t0
        history.append(record)

        val = f" | val {format_metrics(record['val'])}" if "val" in record else ""
        logger.info("epoch %2d | loss %.4f (ce %.4f, kd %.4f) | train top1 %.2f%s | lr %.2e | %.0fs",
                    epoch + 1, record["loss"], record["ce"], record["kd"], record["top1"], val,
                    record["lr"], record["seconds"])
        if main:
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            torch.save({
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "average": average,
                "averaged": averaged,
                "config": cfg.to_dict(),
            }, last)

    # ---- weight averaging -----------------------------------------------
    results = {"averaged_epochs": [t.average_from, t.epochs], "last_epoch": history[-1]}
    if main:
        _section("weight averaging")
        torch.save({
            "model": average,
            "config": cfg.to_dict(),
            "averaged_epochs": [t.average_from, t.epochs],
            "classes": train_set.classnames,
        }, run_dir / "averaged.pt")
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        last.unlink(missing_ok=True)
        logger.info("averaged %d epoch(s) into %s", averaged, run_dir / "averaged.pt")
    dist.barrier()
    dist.cleanup()
    return results
