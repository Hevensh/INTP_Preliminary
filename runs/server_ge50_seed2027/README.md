# GE eight-way results — 2026-09-11

Source: `results_no_weights.zip`, collected_summary.json and each GE run's summary.json, metrics.jsonl, config.json, environment.json and console.log. No weights were supplied.

All eight GE runs completed 50 epochs. Independently checked that history contains epochs 1–50 exactly once, equals metrics.jsonl, and has 129536 training / 5000 validation samples every epoch. Recomputed best validation Top-1 agrees with summary.json.

Common controls: seed 2027, batch 512, DALI, AdamW LR 0.0005, warmup 5, then constant LR, weight decay 0.05, V100-SXM2-32GB. This is the small explicit-direction GE branch, not the previous ~5.5M cos/sin-to-ViT branch.

| Lattice | Directions | Scales | Parameters | Best Top-1 % | Epoch | Top-5 at that epoch % | Final Top-1 % |
|---|---|---|---:|---:|---:|---:|---:|
| Square | Full 4 | Single | 1,336,956 | 61.44 | 47 | 85.36 | 60.72 |
| Square | Full 4 | Multi | 1,336,956 | 59.24 | 48 | 84.02 | 59.00 |
| Square | Half 2 | Single | 1,336,956 | 60.78 | 50 | 85.18 | 60.78 |
| Square | Half 2 | Multi | 1,336,956 | 59.74 | 47 | 84.08 | 59.18 |
| Hex | Full 6 | Single | 1,328,652 | 63.08 | 48 | 86.70 | 62.62 |
| Hex | Full 6 | Multi | 1,328,652 | 61.92 | 50 | 86.04 | 61.92 |
| Hex | Half 3 | Single | 1,328,652 | 62.02 | 48 | 86.00 | 61.70 |
| Hex | Half 3 | Multi | 1,328,652 | 60.70 | 50 | 85.76 | 60.70 |

## Interpretation

- Multi minus single best Top-1: Square full -2.20 pp, Square half -1.04 pp, Hex full -1.16 pp, Hex half -1.32 pp. Final-epoch differences have the same sign. This implemented multi-scale configuration consistently loses here; it does not establish that multi-scale methods generally hurt.
- Full minus half: Square single +0.66 pp, Square multi -0.50 pp; Hex single +1.06 pp, Hex multi +1.22 pp. Half directions are not universally better in this explicit-direction architecture.
- Hex minus Square matched full/half and scale labels: +1.64, +2.68, +1.24, +0.96 pp. Direction counts, front-end kernels and neighborhood/downsampling differ, so this is not an isolated causal effect of the lattice.
- Best Hex is 63.08%, compared with previous ordinary ViT 64.70% in the same supplied archive, but 1.329M vs 5.544M parameters and different backbone structure. These are not parameter-matched models.
- Single seed only; best checkpoint is selected on validation. No statistical significance claim.

## Runtime and continuation

Median seconds per epoch includes train + validation. Old records are explicitly labelled unmarked, not inferred from the latest environment snapshot.

| Run | Old/unmarked epochs | Old median seconds | Triton epochs | Triton median seconds |
|---|---|---:|---|---:|
| Square full single | 1–35 | 280.17 | 36–50 | 129.52 |
| Square full multi | 1–35 | 281.93 | 36–50 | 129.29 |
| Square half single | 1–50 | 154.99 | None recorded | — |
| Square half multi | 1–50 | 155.42 | None recorded | — |
| Hex full single | 1–7 | 294.93 | 8–50 | 154.37 |
| Hex full multi | 1–7 | 299.27 | 8–50 | 156.50 |
| Hex half single | — | — | 1–50 | 115.47 |
| Hex half multi | — | — | 1–50 | 115.93 |

Observed full-run before/after speed ratios: Square ~2.16–2.18x; Hex ~1.91x. These are real recorded wall-time changes, not an isolated kernel benchmark; concurrent server load may contribute. Square half runs have no new-backend marker and should not be used to compare new full vs new half performance.

Resume console logs report restored model/optimizer/scheduler/scaler/RNG at epoch 36 for Square full and epoch 8 for Hex full. Epoch sequences are continuous. Without checkpoints in this ZIP, internal optimizer tensors cannot be independently audited here.

Do not interpret `last_session_wall_minutes` as total 50-epoch cost for resumed runs. The archive's `epoch_compute_minutes` sums recorded train/validation epochs, excluding interrupted partial epochs and other overhead.


## Published artifacts

Each run includes portable config.json, all 50 metrics.jsonl records, model_summary.json, allowlisted runtime.json and resume.json. collected_summary.json includes only these eight completed runs. provenance.json records the original ZIP and source-member SHA256 checksums. Raw datasets, checkpoints, caches, full console logs and machine-specific paths are excluded. JSON formatting and line endings may be normalized; metric values are unchanged.

Dataset: [ambityga/imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100), version 8; follow the upstream data license. This is a result archive, not a complete training-code snapshot. The result commit must not be cited as the original training-code revision.
