# Multi-probe Image / Feature Look

This opt-in experiment preserves old M=1 checkpoints and the established
half4d4r path. The new grid-first path currently supports half6d3r with a 4x12
Look grid. It does not silently convert old independent Feature direction W.

## Computation

Each head has M independently initialized probes, each with its own null score
and output template. Feature probes are shared within the existing G-layer
groups; output templates remain independent by layer/head/probe. Image probes
are independent by layer/head/probe. Existing feature extraction timing is
unchanged (Feature probes consume initial paired token features).

Feature directions share a canonical paired weight W. With J(a,b)=(-b,a):

```
u = x dot W; v = x dot JW
score[d] = u*cos(theta[d]) + v*sin(theta[d]) + shared_bias
p[m] = softmax(concat(score[m], null[m]))[..., real_poses]
```

There is one shared direction bias per probe, not independent offsets per
direction. Null has zero output and its probability is NOT renormalized away.
Each probe independently normalizes across its poses plus null; probes do not
compete through another softmax. Image probes normalize over both scales and
angles plus null.

For each scale, first form a query-conditioned small grid:

```
grid[s] = sum_{m,d} p[m,s,d] * rotate(template[m], theta[d]) / M
bias = sum_s interpolate_s(grid[s])
```

The average keeps M from multiplying bias magnitude; it is not a learned gate.
Each of the M templates remains independently learnable. Rotation is an exact
integer angular-bin roll on this grid (12 bins / 12-direction full period).
Distinct scale supports still require distinct sampling. Feature Look's one
grid and Image Look's larger-scale grid have identical sampling coordinates,
so they are added before interpolation. Thus dual Look performs **two spatial
samplings per layer**, independent of M and of the six matched poses. No tensor
of shape `[heads,M,scales,poses,queries,keys]` is created.

The existing Triton dense-grid sampler and structured attention are reused;
the latter receives the dense bias plus a zero singleton structured term.
Dense per-query bias is still stored, so memory grows with batch and token
count; this is not a promise of constant-memory attention.

## Local performance (2026-09-05)

RTX 4060 Laptop 8 GiB, B=128, 224x224, 195 patch tokens, FP16 AMP, G=3,
undifferentiated tokenizer. Three warmups, median of 30 repeats, no dataloader,
optimizer step/state, DDP, or compile time. Random synthetic inputs, nonzero
Look templates to exercise probe gradients. All gradients finite.

| Image M / Feature M | Parameters | Forward + backward (ms) | Inference (ms) | Peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Legacy 1 / 1 (independent Feature direction W) | 5,496,868 | 401.34 | 101.32 | 3,072.42 |
| Rotating W, grid-first 1 / 1 | 5,492,968 | 374.35 | 99.51 | 3,126.75 |
| Rotating W, grid-first 1 / 4 | 5,500,096 | 388.85 | 102.25 | 3,180.65 |
| Rotating W, grid-first 4 / 4 | 5,581,204 | 481.33 | 111.73 | 3,373.49 |

Against the same optimized 1/1 path, 4/4 adds about 29% forward/backward time,
12% inference time and 8% peak allocated memory. Against the legacy path, the
corresponding increases are about 20%, 10%, 10%. These are local synthetic
measurements, NOT T4 throughput or ImageNet accuracy. B=32 tests showed a larger
benefit from the rewrite; do not extrapolate small-batch timings to Kaggle.

Raw local results: `runs/multiprobe_benchmark_b128_final.json` (ignored artifact).
Reproduce on the target machine:

```bash
python -m experiments.imagenet100.benchmark_multiprobe_look --batch-size 128 --steps 30
```

## Training

Keep PE + dual Look, G=3, half6d3r, K24/K12, 96 initial Full prototypes,
differentiation at epochs 3/5/7 retaining 3/4, 1/2, 1/4 Full, no new Color
candidates. Batch256/GPU, two GPUs, 20 epochs. New names avoid overwriting
prior runs. Both probe counts and rotation mode are saved in checkpoint config
and supported by the evaluation/geometry-export builders.

Kaggle cells (one experiment at a time):

```bash
!bash scripts/kaggle/run_imagenet100_dual_look_multiprobe_2xt4.sh 1 4
!bash scripts/kaggle/run_imagenet100_dual_look_multiprobe_2xt4.sh 4 4
```

The differentiated runner requires at least seven epochs, because its last
split is at epoch seven. It deliberately rejects EPOCHS=5 instead of silently
changing that schedule. No full training or accuracy evaluation was run locally.

## Validation

71 targeted tests pass: explicit rotated-W vs two-dot-product responses and
input gradients; independent interpolation vs grid-first values and gradients;
both Image scales; null retention; merging Image/Feature small grids; existing
sampler, attention, model factory, training and differentiation regression tests.
CUDA full-model forward/backward tested for all four benchmark configurations.

## Polar W storage check

For each paired weight `W=(r*cos(phi), r*sin(phi))`, rotating W by theta is
exactly `r*(cos(phi+theta), sin(phi+theta))`. Factoring this into two projections
gives the same scores. Float64 output and x/r/phi gradient checks passed.

Probe-only B128/G3/M4 forward-backward microbenchmark (40 repeats, two rounds):
Cartesian W 5.90/5.74 ms; polar storage with one reconstruction 5.94/6.11 ms.
No speed benefit was observed; the default stays Cartesian. Both store two
scalars per pair. The large per-direction W is already eliminated by the
two-projection identity. Directly evaluating input/weight angle differences
would additionally need input magnitudes/phases or expanded trigonometry.

Polar and Cartesian parameterizations also have different optimizer dynamics;
Cartesian Adam moments cannot simply be reinterpreted as r/phi moments.
No existing parameter or optimizer state was converted.

```bash
python -m experiments.imagenet100.benchmark_look_probe_storage
```
