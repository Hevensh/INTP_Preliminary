from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn

from experiments.imagenet100.data import build_imagefolder_loaders, discover_imagefolder_splits


@dataclass
class TrainConfig:
    experiment_name: str = "deit_tiny_imagenet100_smoke"
    data_root: str = "/kaggle/input/imagenet100"
    output_root: str = "/kaggle/working/runs"
    model: str = "deit_tiny_patch16_224"
    num_classes: int = 100
    image_size: int = 224
    pretrained: bool = False
    epochs: int = 5
    batch_size: int = 128
    num_workers: int = 4
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 0
    amp: bool = True
    device: str = "auto"
    train_samples: int | None = None
    val_samples: int | None = None
    max_train_steps: int | None = None
    max_val_steps: int | None = None
    progress_interval_seconds: float = 60.0
    resume: str | None = None


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    top1: float
    top5: float
    samples: int
    seconds: float
    samples_per_second: float


def _load_config(path: Path) -> TrainConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    known = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return TrainConfig(**raw)


def _validate_config(config: TrainConfig) -> None:
    if min(config.epochs, config.batch_size, config.num_classes, config.image_size) <= 0:
        raise ValueError("epochs, batch_size, num_classes, and image_size must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "timm": timm.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "git_commit": _git_commit(),
    }
    if device.type == "cuda":
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count": torch.cuda.device_count(),
                "gpu_total_memory_mib": round(
                    torch.cuda.get_device_properties(device).total_memory / 1024**2, 2
                ),
            }
        )
    return result


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _cosine_lambda(step: int, *, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return max((step + 1) / max(warmup_steps, 1), 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predictions = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    matches = predictions.eq(targets[:, None])
    return int(matches[:, :1].sum()), int(matches.sum())


def _run_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float,
    max_steps: int | None,
    epoch: int,
    phase: str,
    progress_interval_seconds: float,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = total_samples = top1_correct = top5_correct = 0
    started = time.perf_counter()
    next_progress_at = started + progress_interval_seconds
    total_steps = min(len(loader), max_steps if max_steps is not None else len(loader))
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, (images, targets) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if optimizer is not None:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            count = int(targets.numel())
            correct1, correct5 = _topk_correct(logits.detach(), targets)
            total_loss += float(loss.detach()) * count
            total_samples += count
            top1_correct += correct1
            top5_correct += correct5
            now = time.perf_counter()
            if progress_interval_seconds > 0 and now >= next_progress_at:
                elapsed = now - started
                completed_steps = step + 1
                estimated_total = elapsed * total_steps / completed_steps
                progress: dict[str, Any] = {
                    "event": "progress",
                    "phase": phase,
                    "epoch": epoch,
                    "step": completed_steps,
                    "steps": total_steps,
                    "progress_percent": round(100.0 * completed_steps / total_steps, 2),
                    "loss": total_loss / total_samples,
                    "top1": top1_correct / total_samples,
                    "learning_rate": optimizer.param_groups[0]["lr"] if optimizer is not None else None,
                    "samples_per_second": total_samples / elapsed,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": max(estimated_total - elapsed, 0.0),
                }
                if device.type == "cuda":
                    progress["peak_cuda_allocated_mib"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                    )
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                intervals_elapsed = math.floor(elapsed / progress_interval_seconds) + 1
                next_progress_at = started + intervals_elapsed * progress_interval_seconds
    if total_samples == 0:
        raise RuntimeError("epoch processed zero samples")
    seconds = time.perf_counter() - started
    return EpochMetrics(
        loss=total_loss / total_samples,
        top1=top1_correct / total_samples,
        top5=top5_correct / total_samples,
        samples=total_samples,
        seconds=seconds,
        samples_per_second=total_samples / seconds,
    )


def _checkpoint_payload(
    *,
    epoch: int,
    best_top1: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "best_top1": best_top1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_top1", -1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible DeiT-Tiny ImageNet-100 training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = _load_config(args.config)
    for name in ("data_root", "output_root", "epochs", "batch_size", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    _validate_config(config)
    _seed_everything(config.seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = bool(config.amp and device.type == "cuda")

    splits = discover_imagefolder_splits(config.data_root, expected_classes=config.num_classes)
    train_loader, val_loader, full_train_size, full_val_size = build_imagefolder_loaders(
        splits,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        train_samples=config.train_samples,
        val_samples=config.val_samples,
    )
    model = timm.create_model(
        config.model,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        img_size=config.image_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = min(
        len(train_loader),
        config.max_train_steps if config.max_train_steps is not None else len(train_loader),
    )
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = round(steps_per_epoch * config.warmup_epochs)
    min_ratio = config.min_learning_rate / config.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_lambda(
            step, total_steps=total_steps, warmup_steps=warmup_steps, min_ratio=min_ratio
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_dir = Path(config.output_root) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    _atomic_json(run_dir / "config.json", asdict(config))
    environment = _environment(device)
    environment["dataset"] = {
        "requested_root": str(Path(config.data_root).resolve()),
        "train_roots": [str(path) for path in splits.train],
        "val_root": str(splits.val),
        "classes": len(splits.classes),
        "full_train_images": full_train_size,
        "full_val_images": full_val_size,
        "effective_train_images": len(train_loader.dataset),
        "effective_val_images": len(val_loader.dataset),
    }
    _atomic_json(run_dir / "environment.json", environment)
    model_summary = {
        "name": config.model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "image_size": config.image_size,
        "num_classes": config.num_classes,
    }
    _atomic_json(run_dir / "model_summary.json", model_summary)
    print(json.dumps({"event": "setup", **environment["dataset"], **model_summary}, ensure_ascii=False))

    start_epoch, best_top1 = 1, -1.0
    if config.resume is not None:
        start_epoch, best_top1 = _restore_checkpoint(
            Path(config.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip_norm=config.grad_clip_norm,
            max_steps=config.max_train_steps,
            epoch=epoch,
            phase="train",
            progress_interval_seconds=config.progress_interval_seconds,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            optimizer=None,
            scheduler=None,
            scaler=scaler,
            grad_clip_norm=config.grad_clip_norm,
            max_steps=config.max_val_steps,
            epoch=epoch,
            phase="val",
            progress_interval_seconds=config.progress_interval_seconds,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(record)
        _append_jsonl(metrics_path, record)
        improved = val_metrics.top1 > best_top1
        best_top1 = max(best_top1, val_metrics.top1)
        payload = _checkpoint_payload(
            epoch=epoch,
            best_top1=best_top1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
        )
        _save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            _save_checkpoint(run_dir / "best.pt", payload)
        print(json.dumps(record, ensure_ascii=False))

    summary: dict[str, Any] = {
        "status": "complete",
        "experiment_name": config.experiment_name,
        "best_val_top1": best_top1,
        "completed_epochs": config.epochs,
        "wall_seconds": time.perf_counter() - training_started,
        "history": history,
        **model_summary,
    }
    if device.type == "cuda":
        summary.update(
            {
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            }
        )
    _atomic_json(run_dir / "summary.json", summary)
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
