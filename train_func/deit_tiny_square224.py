from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from train_func.common import set_seed
from train_func.deit_tiny_hex224 import build_deit_tiny_source, build_loaders, run_epoch, set_epoch_cosine_lr


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen standard DeiT-Tiny CIFAR-10 transfer baseline.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--head-output", type=Path)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    amp = device.type == "cuda" and not args.no_amp

    model = build_deit_tiny_source(pretrained=True, checkpoint=args.checkpoint)
    model.reset_classifier(10)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
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
    history: list[dict[str, object]] = []
    best_test_accuracy = -1.0
    best_epoch = 0
    best_head_state: dict[str, torch.Tensor] | None = None
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
            best_head_state = {
                key: value.detach().cpu().clone()
                for key, value in model.head.state_dict().items()
            }
        print(json.dumps(record, ensure_ascii=False))

    result: dict[str, object] = {
        "model": "deit_tiny_patch16_224_square_frozen",
        "trainable": "classification_head_only",
        "device": str(device),
        "amp": amp,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "image_size": 224,
        "patch_size": 16,
        "patches": 196,
        "sequence_length": 197,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head_lr": args.head_lr,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_ratio": args.min_lr_ratio,
        "best_epoch": best_epoch,
        "best_test_accuracy": best_test_accuracy,
        "history": history,
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
    print(json.dumps(result, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.head_output is not None:
        if best_head_state is None:
            raise RuntimeError("no best head state was recorded")
        args.head_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_head_state, args.head_output)


if __name__ == "__main__":
    main()
