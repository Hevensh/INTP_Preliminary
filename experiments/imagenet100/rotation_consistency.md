# Rotation-consistency probe

This local diagnostic measures prediction stability after rotating the
ImageNet-100 validation images. It is a bounded probe, not a replacement for
the aligned validation-accuracy table and not evidence of exact equivariance.

## Protocol

- Data: 1,000 validation images, deterministically balanced as 10 images from
  each of the 100 classes.
- Angles: 0 through 345 degrees in 15-degree increments.
- Exact multiples of 90 degrees use `torch.rot90`. Other angles use reflection
  padding, bilinear rotation, and a center crop before ImageNet normalization.
- Agreement compares each rotated prediction with that model's own 0-degree
  prediction. JSD is the Jensen-Shannon divergence from the same reference.
- All checkpoints are loaded with `strict=True`; no direction-grid
  interpolation or legacy-shape compatibility is used.

## Preliminary result

| Model | 0-degree Top-1 | Mean rotated Top-1 | Mean drop | Worst Top-1 | Mean agreement | Mean JSD | Cardinal Top-1 | Cardinal drop | Cardinal agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard DeiT-Tiny | 51.60 | 41.11 | 10.49 | 35.80 | 53.16 | 0.1136 | 39.77 | 11.83 | 52.80 |
| SHARE half4d4r, PE + Image Look | 53.40 | 41.77 | 11.63 | 34.90 | 54.62 | 0.1200 | 40.67 | 12.73 | 52.63 |
| SHARE half6d3r, PE + Image Look + G3 Feature Look | 55.00 | 43.44 | 11.56 | 37.30 | 54.28 | 0.1218 | 42.00 | 13.00 | 52.17 |
| GE-ViT p4 local | 58.10 | 52.08 | 6.02 | 47.20 | 67.38 | 0.0744 | 56.83 | 1.27 | 82.87 |

`Mean rotated` excludes the 0-degree reference. `Cardinal` contains 90, 180,
and 270 degrees. The balanced subset can have a different 0-degree accuracy
from the full 5,000-image validation set, so the diagnostic should compare
stability metrics within this table rather than substitute these numbers into
the main accuracy table.

The result supports a conservative distinction. SHARE improves absolute
classification under many rotated views because its unrotated classifier is
stronger, but its relative stability is close to the standard ViT. GE-ViT
retains an explicit C4 orientation field throughout the backbone and is much
more stable at 90-degree group rotations. Therefore the current SHARE evidence
supports local pose sharing and pose-conditioned attention, not end-to-end
rotation equivariance.

## Reproduction

Use `experiments.imagenet100.eval_rotation_consistency` with
`--samples 1000 --angles 0:345:15`. The local JSON artifacts are stored below
`runs/rotation_consistency/` and are intentionally ignored by Git together
with model weights and downloaded validation images.
