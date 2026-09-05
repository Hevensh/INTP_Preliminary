from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from experiments.imagenet100.data import (
    build_imagefolder_loaders,
    discover_imagefolder_splits,
    load_or_index_imagefolder_samples,
)
from experiments.imagenet100.models import MODEL_VARIANTS, build_imagenet100_model
from experiments.imagenet100.differentiation_optimizer import (
    rebuild_adamw_and_scheduler,
)


@dataclass
class TrainConfig:
    experiment_name: str = "deit_tiny_imagenet100_smoke"
    data_root: str = "/kaggle/input/imagenet100"
    output_root: str = "/kaggle/working/runs"
    model: str = "deit_tiny_patch16_224"
    model_variant: str = "deit_tiny"
    num_classes: int = 100
    image_size: int = 224
    pretrained: bool = False
    epochs: int = 5
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
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
    hex_kernel_size: int = 21
    hex_stride: int = 18
    rot_kernel_sizes: tuple[int, ...] = (24, 12)
    rot_bases: int = 96
    rot_directions: int = 4
    rot_global_directions: int = 8
    rot_angular_bins_per_radius: int = 4
    look_compact_variable_rings: bool = False
    center_look_layers_per_probe: int = 1
    image_look_probes: int = 1
    feature_look_probes: int = 1
    feature_look_rotating_probes: bool = False
    sparse_hex_look: bool = False
    feature_ring_look: bool = False
    feature_ring_start_layer: int = 0
    feature_ring_group_size: int = 4
    feature_ring_frequency: bool = False
    rot_prototype_chunk_size: int = 16
    rot_use_null: bool = True
    rot_null_initial_score: float = -1.0
    rot_score_normalization: str = "none"
    rot_response_gate: str = "exp2"
    rot_response_gate_location: str = "pose"
    rot_score_clamp: float = 4.0
    rot_progressive_differentiation: bool = False
    rot_differentiation_epochs: tuple[int, ...] = (3, 5, 7)
    rot_full_retention_fractions: tuple[float, ...] = (0.75, 0.50, 0.25)
    rot_family_complexity_weight: float = 0.4
    rot_stripe_longitudinal_bins: int = 3
    rot_stripe_offset_subdivisions: int = 4
    gmr_hidden_channels: int = 24
    arc_kernel_number: int = 4
    arc_max_angle_degrees: float = 40.0
    arc_batch_chunk_size: int = 32


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


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
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if config.model_variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant must be one of {sorted(MODEL_VARIANTS)}")
    if min(config.hex_kernel_size, config.hex_stride) <= 0:
        raise ValueError("hex_kernel_size and hex_stride must be positive")
    if config.rot_angular_bins_per_radius <= 0:
        raise ValueError("rot_angular_bins_per_radius must be positive")
    if config.center_look_layers_per_probe <= 0:
        raise ValueError("center_look_layers_per_probe must be positive")
    if min(config.image_look_probes, config.feature_look_probes) <= 0:
        raise ValueError("Look probe counts must be positive")
    if min(
        config.gmr_hidden_channels,
        config.arc_kernel_number,
        config.arc_batch_chunk_size,
    ) <= 0:
        raise ValueError("GMR width, ARC kernel count, and ARC chunk size must be positive")
    if config.arc_max_angle_degrees <= 0:
        raise ValueError("arc_max_angle_degrees must be positive")
    if config.rot_progressive_differentiation:
        epochs = tuple(int(epoch) for epoch in config.rot_differentiation_epochs)
        fractions = tuple(float(value) for value in config.rot_full_retention_fractions)
        if len(epochs) != len(fractions) or not epochs:
            raise ValueError(
                "rot_differentiation_epochs and rot_full_retention_fractions "
                "must have the same non-zero length"
            )
        if any(epoch <= 0 or epoch > config.epochs for epoch in epochs):
            raise ValueError("differentiation epochs must lie inside the training schedule")
        if any(right <= left for left, right in zip(epochs, epochs[1:])):
            raise ValueError("differentiation epochs must be strictly increasing")
        if any(not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError("Full retention fractions must lie in (0, 1]")
        if any(right >= left for left, right in zip(fractions, fractions[1:])):
            raise ValueError("Full retention fractions must be strictly decreasing")
        if len(tuple(config.rot_kernel_sizes)) != 2:
            raise ValueError("progressive differentiation currently requires two scales")
        if not config.rot_use_null:
            raise ValueError("progressive differentiation requires null-softmax routing")
        if config.rot_family_complexity_weight < 0:
            raise ValueError("rot_family_complexity_weight must be non-negative")
        if min(
            config.rot_stripe_longitudinal_bins,
            config.rot_stripe_offset_subdivisions,
        ) <= 0:
            raise ValueError("Stripe resolutions must be positive")


def _distributed_context(requested_device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if requested_device not in {"auto", "cuda"}:
            raise ValueError("distributed execution currently requires CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun requested distributed CUDA, but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30),
        )
        return DistributedContext(rank, local_rank, world_size, torch.device("cuda", local_rank))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    return DistributedContext(rank=0, local_rank=0, world_size=1, device=device)


def _raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


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


def _validate_cuda_architecture(device: torch.device) -> None:
    """Fail early when the Kaggle torch build cannot execute on its GPU."""

    if device.type != "cuda":
        return
    capability = torch.cuda.get_device_capability(device)
    compiled_arches = torch.cuda.get_arch_list()
    supported = sorted(
        {
            (int(arch[3:]) // 10, int(arch[3:]) % 10)
            for arch in compiled_arches
            if arch.startswith("sm_") and arch[3:].isdigit()
        }
    )
    if supported and capability < supported[0]:
        gpu_name = torch.cuda.get_device_name(device)
        formatted = " ".join(f"sm_{major}{minor}" for major, minor in supported)
        raise RuntimeError(
            f"GPU {gpu_name} has CUDA capability sm_{capability[0]}{capability[1]}, "
            f"but this PyTorch build starts at sm_{supported[0][0]}{supported[0][1]} "
            f"(compiled: {formatted}). On Kaggle, select a T4/L4 or newer GPU and "
            "restart the session. Do not reinstall the preloaded torch stack."
        )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _progress_line(progress: dict[str, Any], *, epochs: int) -> str:
    memory = progress.get("peak_cuda_allocated_mib")
    memory_text = "" if memory is None else f" | mem {memory / 1024:.1f}G"
    return (
        f"[{progress['phase']} {progress['epoch']:02d}/{epochs:02d}] "
        f"{progress['progress_percent']:5.1f}% "
        f"({progress['step']}/{progress['steps']})"
        f" | loss {progress['loss']:.3f}"
        f" | top1 {100 * progress['top1']:.1f}%"
        f" | {progress['samples_per_second']:.0f} img/s"
        f" | eta {_duration(progress['eta_seconds'])}"
        f"{memory_text}"
    )


def _epoch_line(epoch: int, train: EpochMetrics, val: EpochMetrics) -> str:
    return (
        f"[epoch {epoch:02d}] "
        f"train loss {train.loss:.3f}, top1 {100 * train.top1:.2f}%"
        f" | val loss {val.loss:.3f}, top1 {100 * val.top1:.2f}%, "
        f"top5 {100 * val.top5:.2f}%"
        f" | {_duration(train.seconds + val.seconds)}"
    )


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
    gradient_accumulation_steps: int,
    max_steps: int | None,
    epoch: int,
    epochs: int,
    phase: str,
    progress_interval_seconds: float,
    distributed: DistributedContext,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = total_samples = top1_correct = top5_correct = 0
    started = time.perf_counter()
    next_progress_at = started + progress_interval_seconds
    total_steps = min(len(loader), max_steps if max_steps is not None else len(loader))
    context = torch.enable_grad if training else torch.no_grad
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    with context():
        for step, (images, targets) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            update_now = (
                optimizer is not None
                and (
                    (step + 1) % gradient_accumulation_steps == 0
                    or step + 1 == total_steps
                )
            )
            sync_context = (
                model.no_sync()
                if training
                and isinstance(model, DistributedDataParallel)
                and not update_now
                else nullcontext()
            )
            with sync_context:
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    logits = model(images)
                    loss = criterion(logits, targets)
                if optimizer is not None:
                    group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
                    group_size = min(
                        gradient_accumulation_steps,
                        total_steps - group_start,
                    )
                    scaler.scale(loss / group_size).backward()
            if update_now:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            count = int(targets.numel())
            correct1, correct5 = _topk_correct(logits.detach(), targets)
            total_loss += float(loss.detach()) * count
            total_samples += count
            top1_correct += correct1
            top5_correct += correct5
            now = time.perf_counter()
            if distributed.is_main and progress_interval_seconds > 0 and now >= next_progress_at:
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
                    "samples_per_second": total_samples / elapsed * distributed.world_size,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": max(estimated_total - elapsed, 0.0),
                }
                if device.type == "cuda":
                    progress["peak_cuda_allocated_mib"] = (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                    )
                print(_progress_line(progress, epochs=epochs), flush=True)
                intervals_elapsed = math.floor(elapsed / progress_interval_seconds) + 1
                next_progress_at = started + intervals_elapsed * progress_interval_seconds
    if total_samples == 0:
        raise RuntimeError("epoch processed zero samples")
    seconds = time.perf_counter() - started
    if distributed.enabled:
        totals = torch.tensor(
            [total_loss, total_samples, top1_correct, top5_correct],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        elapsed = torch.tensor(seconds, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        total_loss, total_samples, top1_correct, top5_correct = totals.tolist()
        seconds = float(elapsed)
        total_samples = int(total_samples)
        top1_correct = int(top1_correct)
        top5_correct = int(top5_correct)
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
        "model": _raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": asdict(config),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
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
    _raw_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        cuda_rng = rng["cuda"]
        if isinstance(cuda_rng, (list, tuple)):
            # Compatibility with checkpoints written before DDP stored only
            # the current process' CUDA RNG state.
            cuda_rng = cuda_rng[min(torch.cuda.current_device(), len(cuda_rng) - 1)]
        torch.cuda.set_rng_state(cuda_rng)
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_top1", -1.0))


def _prepare_model_for_checkpoint(model: nn.Module, path: Path) -> None:
    """Restore dynamic prototype shapes before optimizer construction."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    patch_embed = getattr(model, "patch_embed", None)
    prepare = getattr(patch_embed, "prepare_for_state_dict", None)
    if prepare is not None:
        prepare(checkpoint["model"], prefix="patch_embed.")


def _wrap_distributed(model: nn.Module, context: DistributedContext) -> nn.Module:
    if not context.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def _apply_progressive_differentiation(
    *,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    target_full_count: int,
    complexity_weight: float,
    learning_rate: float,
    weight_decay: float,
    scheduler_factory,
    distributed: DistributedContext,
) -> tuple[
    nn.Module,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, Any],
]:
    raw_model = _raw_model(model)
    patch_embed = getattr(raw_model, "patch_embed", None)
    if patch_embed is None or not hasattr(patch_embed, "plan_differentiation"):
        raise TypeError("the selected tokenizer does not support differentiation")

    payload: list[Any] = [None]
    if distributed.is_main:
        payload[0] = patch_embed.plan_differentiation(
            target_full_count=target_full_count,
            complexity_weight=complexity_weight,
        )
    if distributed.enabled:
        dist.broadcast_object_list(payload, src=0, device=distributed.device)
        dist.barrier()
    audit, conversions = patch_embed.apply_differentiation(payload[0])
    optimizer, scheduler, optimizer_audit = rebuild_adamw_and_scheduler(
        model=raw_model,
        old_optimizer=optimizer,
        old_scheduler=scheduler,
        conversions=conversions,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_factory=scheduler_factory,
    )
    model = _wrap_distributed(raw_model, distributed)
    audit["optimizer_state"] = optimizer_audit
    audit["model_parameters"] = sum(
        parameter.numel() for parameter in raw_model.parameters()
    )
    if distributed.enabled:
        dist.barrier()
    return model, optimizer, scheduler, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible DeiT-Tiny ImageNet-100 training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--center-look-layers-per-probe", type=int)
    parser.add_argument("--image-look-probes", type=int)
    parser.add_argument("--feature-look-probes", type=int)
    parser.add_argument("--feature-look-rotating-probes", action="store_true", default=None)
    parser.add_argument("--sparse-hex-look", action="store_true", default=None)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = _load_config(args.config)
    for name in (
        "experiment_name", "data_root", "output_root", "epochs", "batch_size",
        "center_look_layers_per_probe", "resume", "image_look_probes",
        "feature_look_probes", "feature_look_rotating_probes", "sparse_hex_look"
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    _validate_config(config)
    distributed = _distributed_context(config.device)
    device = distributed.device
    _seed_everything(config.seed + distributed.rank)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _validate_cuda_architecture(device)
    amp = bool(config.amp and device.type == "cuda")

    if distributed.is_main:
        print(
            f"[start] {config.experiment_name} | {config.epochs} epochs"
            f" | {distributed.world_size} GPU"
            f" | microbatch {config.batch_size}/GPU"
            f" | accumulation {config.gradient_accumulation_steps}"
            f" (effective global "
            f"{config.batch_size * distributed.world_size * config.gradient_accumulation_steps})",
            flush=True,
        )
    indexed_payload: list[Any] = [None]
    index_started = time.perf_counter()
    if distributed.is_main:
        splits = discover_imagefolder_splits(
            config.data_root, expected_classes=config.num_classes
        )
        print(
            f"[data] found {len(splits.classes)} classes; indexing image paths...",
            flush=True,
        )
        default_cache_dir = (
            Path(config.output_root).expanduser().resolve().parent
            / ".intp_image_index_cache"
        )
        cache_dir = os.environ.get("INTP_IMAGE_INDEX_CACHE", str(default_cache_dir))
        train_index, val_index, cache_hit, cache_path = (
            load_or_index_imagefolder_samples(splits, cache_dir=cache_dir)
        )
        print(
            f"[data] index cache {'hit' if cache_hit else 'created'}: {cache_path}",
            flush=True,
        )
        indexed_payload[0] = (splits, train_index, val_index)
    if distributed.enabled:
        # ImageFolder otherwise walks all 135k files independently on every
        # rank. Broadcast one compact manifest instead of doubling Kaggle I/O.
        dist.broadcast_object_list(
            indexed_payload, src=0, device=distributed.device
        )
    splits, train_index, val_index = indexed_payload[0]
    if distributed.is_main:
        print(
            f"[data] indexed and shared paths in "
            f"{time.perf_counter() - index_started:.1f}s",
            flush=True,
        )
    train_loader, val_loader, full_train_size, full_val_size = build_imagefolder_loaders(
        splits,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        distributed_rank=distributed.rank,
        distributed_world_size=distributed.world_size,
        train_index=train_index,
        val_index=val_index,
    )
    if distributed.is_main:
        print(
            f"[data] train {full_train_size:,} | val {full_val_size:,}; building model...",
            flush=True,
        )
    model = build_imagenet100_model(
        variant=config.model_variant,
        model_name=config.model,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        image_size=config.image_size,
        hex_kernel_size=config.hex_kernel_size,
        hex_stride=config.hex_stride,
        rot_kernel_sizes=tuple(config.rot_kernel_sizes),
        rot_bases=config.rot_bases,
        rot_directions=config.rot_directions,
        rot_global_directions=config.rot_global_directions,
        rot_angular_bins_per_radius=config.rot_angular_bins_per_radius,
        look_compact_variable_rings=config.look_compact_variable_rings,
        center_look_layers_per_probe=config.center_look_layers_per_probe,
        image_look_probes=config.image_look_probes,
        feature_look_probes=config.feature_look_probes,
        feature_look_rotating_probes=config.feature_look_rotating_probes,
        sparse_hex_look=config.sparse_hex_look,
        feature_ring_look=config.feature_ring_look,
        feature_ring_start_layer=config.feature_ring_start_layer,
        feature_ring_group_size=config.feature_ring_group_size,
        feature_ring_frequency=config.feature_ring_frequency,
        rot_prototype_chunk_size=config.rot_prototype_chunk_size,
        rot_use_null=config.rot_use_null,
        rot_null_initial_score=config.rot_null_initial_score,
        rot_score_normalization=config.rot_score_normalization,
        rot_response_gate=config.rot_response_gate,
        rot_response_gate_location=config.rot_response_gate_location,
        rot_score_clamp=config.rot_score_clamp,
        rot_progressive_differentiation=config.rot_progressive_differentiation,
        rot_stripe_longitudinal_bins=config.rot_stripe_longitudinal_bins,
        rot_stripe_offset_subdivisions=config.rot_stripe_offset_subdivisions,
        gmr_hidden_channels=config.gmr_hidden_channels,
        arc_kernel_number=config.arc_kernel_number,
        arc_max_angle_degrees=config.arc_max_angle_degrees,
        arc_batch_chunk_size=config.arc_batch_chunk_size,
    )
    if config.resume is not None:
        _prepare_model_for_checkpoint(model, Path(config.resume))
    model = _wrap_distributed(model.to(device), distributed)
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
    optimizer_steps_per_epoch = math.ceil(
        steps_per_epoch / config.gradient_accumulation_steps
    )
    total_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = round(optimizer_steps_per_epoch * config.warmup_epochs)
    min_ratio = config.min_learning_rate / config.learning_rate
    def scheduler_factory(
        target_optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            target_optimizer,
            lr_lambda=lambda step: _cosine_lambda(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_ratio=min_ratio,
            ),
        )

    scheduler = scheduler_factory(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_dir = Path(config.output_root) / config.experiment_name
    if distributed.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed.enabled:
        dist.barrier()
    metrics_path = run_dir / "metrics.jsonl"
    if distributed.is_main:
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
    environment["distributed"] = {
        "world_size": distributed.world_size,
        "backend": dist.get_backend() if distributed.enabled else None,
        "per_device_batch_size": config.batch_size,
        "global_batch_size": config.batch_size * distributed.world_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_global_batch_size": (
            config.batch_size
            * distributed.world_size
            * config.gradient_accumulation_steps
        ),
    }
    if distributed.is_main:
        _atomic_json(run_dir / "environment.json", environment)
    raw_model = _raw_model(model)
    model_summary = {
        "name": config.model,
        "variant": config.model_variant,
        "parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad
        ),
        "image_size": config.image_size,
        "num_classes": config.num_classes,
    }
    if distributed.is_main:
        _atomic_json(run_dir / "model_summary.json", model_summary)
        print(
            f"[model] {config.model_variant} | {model_summary['parameters']:,} params"
            f" | {device}",
            flush=True,
        )

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
    if distributed.is_main and metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("epoch", 0)) < start_epoch:
                history.append(record)
    differentiation_schedule = {
        int(epoch): round(config.rot_bases * float(fraction))
        for epoch, fraction in zip(
            config.rot_differentiation_epochs,
            config.rot_full_retention_fractions,
        )
    } if config.rot_progressive_differentiation else {}
    training_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if distributed.is_main:
            print(f"[epoch {epoch:02d}/{config.epochs:02d}] train", flush=True)
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
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_steps=config.max_train_steps,
            epoch=epoch,
            epochs=config.epochs,
            phase="train",
            progress_interval_seconds=config.progress_interval_seconds,
            distributed=distributed,
        )
        differentiation_audit = None
        if epoch in differentiation_schedule:
            before = _raw_model(model).patch_embed.family_counts()
            model, optimizer, scheduler, differentiation_audit = (
                _apply_progressive_differentiation(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    target_full_count=differentiation_schedule[epoch],
                    complexity_weight=config.rot_family_complexity_weight,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    scheduler_factory=scheduler_factory,
                    distributed=distributed,
                )
            )
            if distributed.is_main:
                after = differentiation_audit["family_counts"]
                state = differentiation_audit["optimizer_state"]
                print(
                    f"[differentiate {epoch:02d}] Full {before['full']}->{after['full']}"
                    f" | Angular {after['angular']} | Stripe {after['stripe']}"
                    f" | Color {after['color']}"
                    f" | optimizer state {100 * state['state_numel_coverage']:.1f}%",
                    flush=True,
                )
        if distributed.is_main:
            print(f"[epoch {epoch:02d}/{config.epochs:02d}] validate", flush=True)
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
            gradient_accumulation_steps=1,
            max_steps=config.max_val_steps,
            epoch=epoch,
            epochs=config.epochs,
            phase="val",
            progress_interval_seconds=config.progress_interval_seconds,
            distributed=distributed,
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        if differentiation_audit is not None:
            record["differentiation"] = differentiation_audit
        if distributed.is_main:
            history.append(record)
            _append_jsonl(metrics_path, record)
        improved = val_metrics.top1 > best_top1
        best_top1 = max(best_top1, val_metrics.top1)
        if distributed.is_main:
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
            print(_epoch_line(epoch, train_metrics, val_metrics), flush=True)
        if distributed.enabled:
            dist.barrier()

    raw_model = _raw_model(model)
    model_summary["parameters"] = sum(
        parameter.numel() for parameter in raw_model.parameters()
    )
    model_summary["trainable_parameters"] = sum(
        parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad
    )
    if hasattr(raw_model.patch_embed, "family_counts"):
        model_summary["geometry_family_counts"] = raw_model.patch_embed.family_counts()
        model_summary["effective_geometry_parameters"] = sum(
            parameter.numel() for parameter in raw_model.patch_embed.prototype_bank
        )
    summary: dict[str, Any] = {
        "status": "complete",
        "experiment_name": config.experiment_name,
        "best_val_top1": best_top1,
        "completed_epochs": config.epochs,
        "wall_seconds": time.perf_counter() - training_started,
        "history": history,
        **model_summary,
    }
    if config.rot_progressive_differentiation:
        summary["differentiation_history"] = [
            {
                "epoch": record["epoch"],
                **record["differentiation"],
            }
            for record in history
            if "differentiation" in record
        ]
    if hasattr(raw_model, "experiment_diagnostics"):
        diagnostics = raw_model.experiment_diagnostics()
        if diagnostics:
            summary["model_diagnostics"] = diagnostics
    if device.type == "cuda":
        summary.update(
            {
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            }
        )
    if distributed.is_main:
        _atomic_json(run_dir / "model_summary.json", model_summary)
        _atomic_json(run_dir / "summary.json", summary)
        print(
            f"[done] {config.experiment_name}"
            f" | best val top1 {100 * best_top1:.2f}%"
            f" | {_duration(summary['wall_seconds'])}"
            f" | {run_dir}",
            flush=True,
        )
    if distributed.enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
