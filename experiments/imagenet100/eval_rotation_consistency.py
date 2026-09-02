from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from experiments.imagenet100.models import build_imagenet100_model
from experiments.imagenet100.train_vit import TrainConfig, _load_config


class RotatedValidationDataset(Dataset):
    def __init__(self, root: Path, *, image_size: int, angle: float) -> None:
        self.dataset = datasets.ImageFolder(root)
        self.image_size = image_size
        self.angle = float(angle) % 360.0
        self.resize = transforms.Resize(256, interpolation=InterpolationMode.BICUBIC)
        self.crop = transforms.CenterCrop(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _rotate(self, image: torch.Tensor) -> torch.Tensor:
        quarter_turns = round(self.angle / 90.0)
        if math.isclose(self.angle, (quarter_turns * 90.0) % 360.0, abs_tol=1e-6):
            return torch.rot90(image, quarter_turns % 4, dims=(-2, -1))
        padding = math.ceil((math.sqrt(2.0) - 1.0) * self.image_size / 2.0) + 2
        padded = F.pad(image, (padding,) * 4, mode="reflect")
        rotated = TF.rotate(
            padded,
            angle=self.angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
        )
        return TF.center_crop(rotated, [self.image_size, self.image_size])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, target = self.dataset[index]
        image = self.to_tensor(self.crop(self.resize(image)))
        image = self.normalize(self._rotate(image))
        return image, target, index


def _parse_angles(raw: str) -> list[float]:
    if ":" in raw:
        start, stop, step = (float(value) for value in raw.split(":"))
        if step <= 0 or stop < start:
            raise ValueError("angle range must be start:stop:positive_step")
        count = math.floor((stop - start) / step + 1e-9) + 1
        return [start + index * step for index in range(count)]
    angles = [float(value) for value in raw.split(",") if value.strip()]
    if not angles:
        raise ValueError("at least one angle is required")
    return angles


def _resolve_val_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    for candidate in (path / "val.X", path):
        class_dirs = (
            [item for item in candidate.iterdir() if item.is_dir()]
            if candidate.is_dir()
            else []
        )
        if len(class_dirs) == 100:
            return candidate
    raise FileNotFoundError(f"expected a 100-class val.X ImageFolder below {path}")


def _build_model(config: TrainConfig, device: torch.device) -> torch.nn.Module:
    return build_imagenet100_model(
        variant=config.model_variant,
        model_name=config.model,
        pretrained=False,
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
        gmr_hidden_channels=config.gmr_hidden_channels,
        arc_kernel_number=config.arc_kernel_number,
        arc_max_angle_degrees=config.arc_max_angle_degrees,
        arc_batch_chunk_size=config.arc_batch_chunk_size,
    ).to(device)


def _load_weights(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state_dict or contain a 'model' state_dict")
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        probabilities.append(logits.float().softmax(dim=-1).cpu())
        targets.append(labels.cpu())
    return torch.cat(probabilities), torch.cat(targets)


def _angle_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    base_probabilities: torch.Tensor,
) -> dict[str, float]:
    prediction = probabilities.argmax(dim=-1)
    base_prediction = base_probabilities.argmax(dim=-1)
    base_correct = base_prediction.eq(targets)
    top5 = probabilities.topk(min(5, probabilities.shape[-1]), dim=-1).indices.eq(
        targets[:, None]
    ).any(dim=-1)
    mixture = 0.5 * (probabilities + base_probabilities)
    eps = torch.finfo(probabilities.dtype).eps
    jsd = 0.5 * (
        (probabilities * ((probabilities + eps).log() - (mixture + eps).log())).sum(dim=-1)
        + (
            base_probabilities
            * ((base_probabilities + eps).log() - (mixture + eps).log())
        ).sum(dim=-1)
    )
    agreement = prediction.eq(base_prediction)
    agreement_on_base_correct = (
        agreement[base_correct].float().mean().item() if base_correct.any() else float("nan")
    )
    return {
        "top1": 100.0 * prediction.eq(targets).float().mean().item(),
        "top5": 100.0 * top5.float().mean().item(),
        "agreement": 100.0 * agreement.float().mean().item(),
        "agreement_on_base_correct": 100.0 * agreement_on_base_correct,
        "js_divergence": jsd.mean().item(),
        "probability_cosine": F.cosine_similarity(
            probabilities, base_probabilities, dim=-1
        ).mean().item(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _aggregate_rows(rows: list[dict[str, float]], *, base_top1: float) -> dict[str, float]:
    return {
        "angles": len(rows),
        "mean_top1": sum(row["top1"] for row in rows) / len(rows),
        "worst_top1": min(row["top1"] for row in rows),
        "mean_top1_drop": base_top1
        - sum(row["top1"] for row in rows) / len(rows),
        "mean_agreement": sum(row["agreement"] for row in rows) / len(rows),
        "mean_js_divergence": sum(row["js_divergence"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rotation consistency locally.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angles", default="0:345:15")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    angles = _parse_angles(args.angles)
    normalized_angles = [angle % 360.0 for angle in angles]
    if 0.0 not in normalized_angles:
        raise ValueError("angles must include 0 degrees as the consistency reference")
    if normalized_angles[0] != 0.0:
        raise ValueError("0 degrees must be the first requested angle")
    if not any(angle != 0.0 for angle in normalized_angles):
        raise ValueError("at least one non-zero angle is required")
    config = _load_config(args.config)
    val_root = _resolve_val_root(args.val_root)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = device.type == "cuda"

    model = _build_model(config, device)
    _load_weights(model, args.checkpoint)
    model.eval()
    base_probabilities = base_targets = None
    angle_rows: list[dict[str, float]] = []
    started = time.perf_counter()

    for angle in angles:
        dataset: Dataset = RotatedValidationDataset(
            val_root, image_size=config.image_size, angle=angle
        )
        if args.samples is not None:
            if args.samples <= 0:
                raise ValueError("samples must be positive")
            dataset = Subset(dataset, range(min(args.samples, len(dataset))))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        probabilities, targets = _predict(model, loader, device=device, amp=amp)
        if angle % 360.0 == 0.0:
            base_probabilities, base_targets = probabilities, targets
        if base_probabilities is None or base_targets is None:
            raise RuntimeError("0-degree angle must be evaluated before other angles")
        if not torch.equal(targets, base_targets):
            raise RuntimeError("validation sample ordering changed between angles")
        row = {"angle_degrees": float(angle)}
        row.update(_angle_metrics(probabilities, targets, base_probabilities))
        angle_rows.append(row)
        print(
            f"[angle {angle:6.1f}] top1 {row['top1']:.2f}% | "
            f"agree {row['agreement']:.2f}% | JSD {row['js_divergence']:.5f}",
            flush=True,
        )

    nonzero = [row for row in angle_rows if row["angle_degrees"] % 360.0 != 0.0]
    base_top1 = next(row["top1"] for row in angle_rows if row["angle_degrees"] % 360.0 == 0.0)
    cardinal = [
        row
        for row in nonzero
        if math.isclose(row["angle_degrees"] % 90.0, 0.0, abs_tol=1e-6)
    ]
    off_grid = [
        row
        for row in nonzero
        if not math.isclose(row["angle_degrees"] % 90.0, 0.0, abs_tol=1e-6)
    ]
    summary: dict[str, Any] = {
        "base_top1": base_top1,
        "all_rotations": _aggregate_rows(nonzero, base_top1=base_top1),
    }
    if cardinal:
        summary["cardinal_rotations"] = _aggregate_rows(cardinal, base_top1=base_top1)
    if off_grid:
        summary["off_grid_rotations"] = _aggregate_rows(off_grid, base_top1=base_top1)
    payload = {
        "protocol": {
            "dataset": "ambityga/imagenet100 val.X",
            "samples": len(base_targets),
            "angles_degrees": angles,
            "cardinal_rotation": "torch.rot90 after Resize(256)+CenterCrop(image_size)",
            "off_grid_rotation": "reflection pad + bilinear rotate + center crop",
            "normalization": "ImageNet mean/std after rotation",
        },
        "model": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "variant": config.model_variant,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_config": asdict(config),
        },
        "summary": summary,
        "angles": angle_rows,
        "seconds": time.perf_counter() - started,
    }
    _write_json(args.output, payload)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
