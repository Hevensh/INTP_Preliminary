from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import timm
import torch
import torch.nn as nn
from huggingface_hub import try_to_load_from_cache
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from data_func.cifar10 import cifar10_imagenet224_transform
from model.deit_tiny_adapter import load_deit_tiny_state_dict
from model.hex_vit_classifier import HexViTClassifier
from train_func.common import set_seed


TrainStage = Literal["head", "tokenizer", "look"]
TokenizerInit = Literal["transferred", "random", "random_norm_matched"]


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int
    seconds: float


def build_deit_tiny_source(*, pretrained: bool, checkpoint: Path | None = None) -> nn.Module:
    source = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    if pretrained:
        cached = str(checkpoint) if checkpoint is not None else try_to_load_from_cache(
            "timm/deit_tiny_patch16_224.fb_in1k",
            "model.safetensors",
        )
        if isinstance(cached, str):
            source.load_state_dict(load_safetensors(cached), strict=True)
        else:
            source = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    return source


def build_hex224_model(
    *,
    pretrained: bool = True,
    tokenizer_init: TokenizerInit = "transferred",
    checkpoint: Path | None = None,
    enable_polar_look: bool = False,
    polar_look_gate_init: float = 0.0,
) -> tuple[HexViTClassifier, int]:
    if tokenizer_init not in {"transferred", "random", "random_norm_matched"}:
        raise ValueError("tokenizer_init must be transferred, random, or random_norm_matched")
    source = build_deit_tiny_source(pretrained=pretrained, checkpoint=checkpoint)
    model = HexViTClassifier(
        img_size=224,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        hex_kernel_size=21,
        hex_stride=18,
        patch_embed_mode="linear",
        position_mode="learned+polar-look" if enable_polar_look else "learned",
        polar_look_strength=1.0 if enable_polar_look else 0.0,
        polar_look_gate_init=polar_look_gate_init,
    )
    report = load_deit_tiny_state_dict(model, source.state_dict())
    if tokenizer_init == "random":
        model.patch_embed.reset_parameters()
    elif tokenizer_init == "random_norm_matched":
        target_norm = model.patch_embed.weight.detach().flatten(1).norm(dim=1, keepdim=True)
        transferred_bias = model.patch_embed.bias.detach().clone()
        model.patch_embed.reset_parameters()
        with torch.no_grad():
            random_weight = model.patch_embed.weight
            random_norm = random_weight.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
            model.patch_embed.weight.mul_((target_norm / random_norm).unsqueeze(-1))
            model.patch_embed.bias.copy_(transferred_bias)
    return model, len(report.copied_keys)


@torch.no_grad()
def load_hex_stage_checkpoint(model: HexViTClassifier, checkpoint: Path) -> str:
    """Load a saved Hex tokenizer/look stage, including legacy projection-only files."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Hex stage checkpoint must contain a mapping")

    if "patch_embed" not in state:
        # Early tokenizer pilots saved the patch projection state_dict directly.
        model.patch_embed.load_state_dict(state, strict=True)
        return "legacy-patch-only"

    model.patch_embed.load_state_dict(state["patch_embed"], strict=True)
    saved_pos = state.get("pos_embed")
    if saved_pos is not None:
        if model.pos_embed is None or tuple(saved_pos.shape) != tuple(model.pos_embed.shape):
            raise ValueError("checkpoint position embedding does not match target model")
        model.pos_embed.copy_(saved_pos.to(model.pos_embed))

    saved_look = state.get("polar_look_layers")
    if saved_look is not None:
        if model.polar_look_layers is None:
            raise ValueError("checkpoint contains polar look fields but target has none")
        model.polar_look_layers.load_state_dict(saved_look, strict=True)
    saved_gate = state.get("polar_look_gate")
    if saved_gate is not None:
        if model.polar_look_gate is None or tuple(saved_gate.shape) != tuple(model.polar_look_gate.shape):
            raise ValueError("checkpoint polar look gate does not match target model")
        model.polar_look_gate.copy_(saved_gate.to(model.polar_look_gate))
    return "full-stage"


def configure_train_stage(
    model: HexViTClassifier,
    stage: TrainStage,
    *,
    train_head: bool = True,
    train_pos_embed: bool = True,
) -> tuple[str, ...]:
    if stage not in {"head", "tokenizer", "look"}:
        raise ValueError("stage must be head, tokenizer, or look")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if train_head:
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    if stage == "tokenizer":
        for parameter in model.patch_embed.parameters():
            parameter.requires_grad_(True)
        if model.pos_embed is not None and train_pos_embed:
            model.pos_embed.requires_grad_(True)
    elif stage == "look":
        if model.polar_look_layers is None or model.polar_look_gate is None:
            raise ValueError("look stage requires enable_polar_look=True")
        for parameter in model.polar_look_layers.parameters():
            parameter.requires_grad_(True)
        model.polar_look_gate.requires_grad_(True)

    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def build_optimizer(
    model: HexViTClassifier,
    stage: TrainStage,
    *,
    tokenizer_lr: float = 1e-4,
    tokenizer_weight_decay: float = 0.05,
    head_lr: float = 1e-3,
    look_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = []
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": 0.0})

    tokenizer_decay: list[nn.Parameter] = []
    tokenizer_no_decay: list[nn.Parameter] = []
    for name, parameter in model.patch_embed.named_parameters():
        if not parameter.requires_grad:
            continue
        (tokenizer_no_decay if name.endswith("bias") else tokenizer_decay).append(parameter)
    if tokenizer_decay:
        groups.append({"params": tokenizer_decay, "lr": tokenizer_lr, "weight_decay": tokenizer_weight_decay})
    if tokenizer_no_decay:
        groups.append({"params": tokenizer_no_decay, "lr": tokenizer_lr, "weight_decay": 0.0})
    if model.pos_embed is not None and model.pos_embed.requires_grad:
        groups.append({"params": [model.pos_embed], "lr": tokenizer_lr, "weight_decay": 0.0})

    if stage == "look":
        if model.polar_look_layers is None or model.polar_look_gate is None:
            raise ValueError("look stage requires a polar look module")
        look_parameters = [
            parameter for parameter in model.polar_look_layers.parameters() if parameter.requires_grad
        ]
        if model.polar_look_gate.requires_grad:
            look_parameters.append(model.polar_look_gate)
        if look_parameters:
            # Prototype magnitude is part of the response design; do not let
            # AdamW impose an artificial shrinkage pressure on these fields.
            groups.append({"params": look_parameters, "lr": look_lr, "weight_decay": 0.0})

    if not groups:
        raise ValueError(f"stage {stage} produced no trainable parameters")
    return torch.optim.AdamW(groups)


def set_epoch_cosine_lr(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> tuple[float, ...]:
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be non-negative and smaller than epochs")
    if epoch <= warmup_epochs:
        scale = 0.1 + 0.9 * (epoch - 1) / max(warmup_epochs - 1, 1)
    else:
        progress = (epoch - warmup_epochs - 1) / max(epochs - warmup_epochs - 1, 1)
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    result: list[float] = []
    for group in optimizer.param_groups:
        base_lr = group.setdefault("base_lr", group["lr"])
        group["lr"] = float(base_lr) * scale
        result.append(float(group["lr"]))
    return tuple(result)


def _subset(dataset: Dataset, count: int | None, seed: int) -> Dataset:
    if count is None or count >= len(dataset):
        return dataset
    if count <= 0:
        raise ValueError("subset count must be positive")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:count])


def build_loaders(
    *,
    root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    train_samples: int | None,
    test_samples: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_data = CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=cifar10_imagenet224_transform(train=True),
    )
    test_data = CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=cifar10_imagenet224_transform(train=False),
    )
    train_data = _subset(train_data, train_samples, seed)
    test_data = _subset(test_data, test_samples, seed + 1)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_data, shuffle=True, **common),
        DataLoader(test_data, shuffle=False, **common),
    )


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
    max_steps: int | None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    started = time.perf_counter()

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, (images, labels) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            count = int(labels.numel())
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_samples += count
    if total_samples == 0:
        raise ValueError("loader produced no samples")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
        seconds=time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 CIFAR-10 transfer for DeiT-Tiny Hex224.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--stage", choices=("head", "tokenizer", "look"), default="tokenizer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--random-source", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hex-stage-checkpoint", type=Path)
    parser.add_argument(
        "--tokenizer-init",
        choices=("transferred", "random", "random_norm_matched"),
        default="transferred",
    )
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--freeze-pos-embed", action="store_true")
    parser.add_argument("--enable-polar-look", action="store_true")
    parser.add_argument("--polar-look-gate-init", type=float, default=0.0)
    parser.add_argument("--tokenizer-lr", type=float, default=1e-4)
    parser.add_argument("--tokenizer-weight-decay", type=float, default=0.05)
    parser.add_argument("--look-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size) <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.stage == "look" and not args.enable_polar_look:
        raise ValueError("--stage look requires --enable-polar-look")
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp

    model, copied_tensors = build_hex224_model(
        pretrained=not args.random_source,
        tokenizer_init=args.tokenizer_init,
        checkpoint=args.checkpoint,
        enable_polar_look=args.enable_polar_look,
        polar_look_gate_init=args.polar_look_gate_init,
    )
    stage_checkpoint_format = None
    if args.hex_stage_checkpoint is not None:
        stage_checkpoint_format = load_hex_stage_checkpoint(model, args.hex_stage_checkpoint)
    if args.head_checkpoint is not None:
        head_state = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
        model.head.load_state_dict(head_state, strict=True)
    trainable_names = configure_train_stage(
        model,
        args.stage,
        train_head=args.head_checkpoint is None,
        train_pos_embed=not args.freeze_pos_embed,
    )
    model = model.to(device)
    optimizer = build_optimizer(
        model,
        args.stage,
        tokenizer_lr=args.tokenizer_lr,
        tokenizer_weight_decay=args.tokenizer_weight_decay,
        look_lr=args.look_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = build_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    initial_test = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=scaler,
        amp=amp,
        max_steps=args.max_steps,
    )
    print(json.dumps({"initial_test": asdict(initial_test)}, ensure_ascii=False))
    history: list[dict[str, object]] = []
    best_test_accuracy = -1.0
    best_epoch = 0
    best_tokenizer_state: dict[str, object] | None = None
    for epoch in range(1, args.epochs + 1):
        learning_rates = set_epoch_cosine_lr(
            optimizer,
            epoch=epoch,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr_ratio=args.min_lr_ratio,
        )
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            max_steps=args.max_steps,
        )
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            amp=amp,
            max_steps=args.max_steps,
        )
        record = {
            "epoch": epoch,
            "learning_rates": learning_rates,
            "train": asdict(train_metrics),
            "test": asdict(test_metrics),
        }
        history.append(record)
        if test_metrics.accuracy > best_test_accuracy:
            best_test_accuracy = test_metrics.accuracy
            best_epoch = epoch
            best_tokenizer_state = {
                "best_epoch": epoch,
                "patch_embed": {
                    key: value.detach().cpu().clone()
                    for key, value in model.patch_embed.state_dict().items()
                },
                "pos_embed": None if model.pos_embed is None else model.pos_embed.detach().cpu().clone(),
                "polar_look_layers": None
                if model.polar_look_layers is None
                else {
                    key: value.detach().cpu().clone()
                    for key, value in model.polar_look_layers.state_dict().items()
                },
                "polar_look_gate": None
                if model.polar_look_gate is None
                else model.polar_look_gate.detach().cpu().clone(),
            }
        print(json.dumps(record, ensure_ascii=False))

    result: dict[str, object] = {
        "model": "deit_tiny_patch16_224_hex",
        "stage": args.stage,
        "tokenizer_init": args.tokenizer_init,
        "device": str(device),
        "amp": amp,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "max_steps": args.max_steps,
        "image_size": 224,
        "hex_kernel_size": 21,
        "hex_stride": 18,
        "patches": model.patch_embed.num_patches,
        "sequence_length": int(model.pos_embed.shape[1]),
        "copied_tensors": copied_tensors,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else "auto-cache",
        "hex_stage_checkpoint": str(args.hex_stage_checkpoint)
        if args.hex_stage_checkpoint is not None
        else None,
        "hex_stage_checkpoint_format": stage_checkpoint_format,
        "head_checkpoint": str(args.head_checkpoint) if args.head_checkpoint is not None else None,
        "head_frozen": args.head_checkpoint is not None,
        "pos_embed_frozen": args.freeze_pos_embed,
        "polar_look_enabled": args.enable_polar_look,
        "polar_look_gate_init": args.polar_look_gate_init,
        "tokenizer_lr": args.tokenizer_lr,
        "tokenizer_weight_decay": args.tokenizer_weight_decay,
        "look_lr": args.look_lr,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_ratio": args.min_lr_ratio,
        "initial_test": asdict(initial_test),
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_names": trainable_names,
        "history": history,
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
    print(json.dumps({key: value for key, value in result.items() if key != "trainable_names"}, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.model_output is not None:
        if best_tokenizer_state is None:
            raise RuntimeError("no best tokenizer state was recorded")
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_tokenizer_state, args.model_output)


if __name__ == "__main__":
    main()
