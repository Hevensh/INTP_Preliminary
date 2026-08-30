# ImageNet-100 on Kaggle

This experiment uses the ImageNet-100 subset published as
`ambityga/imagenet100`. It contains 100 ImageNet classes with 1,300 training
images and 50 validation images per class.

## Kaggle notebook setup

1. Add the Kaggle dataset `ambityga/imagenet100` as an Input.
2. Enable a T4/L4-or-newer GPU and Internet access. The current Kaggle PyTorch
   image does not include `sm_60` kernels for the older Tesla P100; choose T4
   rather than reinstalling PyTorch in the notebook.
3. Clone the repository and run from its root:

```bash
git clone https://github.com/Hevensh/INTP_Preliminary.git INTPcore
cd INTPcore
pip install -q -r requirements-kaggle.txt
bash scripts/kaggle/run_imagenet100_vit_smoke.sh
```

The loader searches below `/kaggle/input/imagenet100`, merges the four training
shards (`train.X1` through `train.X4`) under one global class index, and pairs
them with `val.X`. It also supports an ordinary single `train`/`val` layout and
rejects candidates that do not contain exactly 100 matching classes.

Run artifacts are written to:

```text
/kaggle/working/runs/deit_tiny_imagenet100_smoke_e5/
  config.json
  environment.json
  model_summary.json
  metrics.jsonl
  summary.json
  best.pt
  last.pt
```

`last.pt` contains the model, optimizer, scheduler, AMP scaler, and RNG states.
Resume a stopped run with:

```bash
python -m experiments.imagenet100.train_vit \
  --config configs/imagenet100/deit_tiny_smoke.json \
  --resume /kaggle/working/runs/deit_tiny_imagenet100_smoke_e5/last.pt
```

The five-epoch result is a smoke test for convergence, runtime, and memory. It
is not the final accuracy comparison.

Training does not use a per-batch progress bar. By default it emits one compact
JSON progress record every 60 seconds with the current phase, epoch, batch
percentage, running loss/Top-1, learning rate, throughput, ETA, and peak CUDA
allocation. Change `progress_interval_seconds` in the JSON config if needed.
Startup records are printed immediately before dataset discovery, image-path
indexing, model construction, and each train/validation phase. Indexing the
135,000 Kaggle-mounted files may take several minutes, but it is no longer a
silent step.
## Two-T4 DeiT versus Hex patch comparison

The 20-epoch comparison uses the same DeiT-Tiny backbone, augmentation,
optimizer, schedule, per-device batch size, and global batch size. The Hex
variant only replaces the ordinary 16x16/stride-16 Conv2d patch embedding with
a 21x21 circular Hex sampler on a stride-18 lattice (195 rather than 196 image
tokens).

From the repository root in a Kaggle notebook with two T4 GPUs:

```bash
DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
bash scripts/kaggle/run_imagenet100_deit_vs_hex_2xt4_e20.sh
```

The default is batch 256 per GPU, hence global batch 512. To lower memory:

```bash
BATCH_SIZE=192 DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
bash scripts/kaggle/run_imagenet100_deit_vs_hex_2xt4_e20.sh
```

Set `RUN_BASELINE=0` or `RUN_HEX=0` to run only one arm. An interrupted arm
automatically resumes from its `last.pt`; a completed arm is skipped. Use a new
`OUTPUT_ROOT` for an intentional fresh rerun.

Each run stores its resolved config, environment, model summary, per-epoch
metrics, best checkpoint, last checkpoint, and final summary below
`/kaggle/working/runs` by default.

## ResNet-18 MAMS branch

This branch tests whether the geometric operator transfers beyond a ViT
tokenizer. The standard arm is torchvision ResNet-18. In the MAMS arm, every
two-3x3 BasicBlock is replaced as a whole by:

```text
MAMS(D6/D3, half4d4r, null-softmax, cos/sin)
  -> BatchNorm -> ReLU -> 1x1 Conv -> BatchNorm -> residual add -> ReLU
```

The ordinary Cartesian feature grid, stem, stage widths, stride-2 shortcuts,
global pooling, and classifier are retained. Diameter 6 uses a pixel-centred
7x7 bounding window with a radius-3 circular support; diameter 3 uses a 3x3
support at the same centers. Prototype initialization is scaled by each stage's
input fan-in so deeper blocks do not receive sharper pose logits solely because
they have more channels. A local AMP forward/backward probe used 2.20 GiB at
batch 32 on an 8-GiB RTX 4060; the Kaggle default is therefore 128/GPU on each
16-GiB T4, with `BATCH_SIZE=64` as the conservative fallback.

Five-epoch smoke runs:

```bash
DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
  bash scripts/kaggle/run_imagenet100_resnet18_2xt4.sh

DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
  bash scripts/kaggle/run_imagenet100_resnet18_mams_4d4r_d6d3_2xt4.sh
```

For the aligned 20-epoch comparison, prepend `EPOCHS=20`. The experiment name
is derived from `EPOCHS`, so the five-epoch smoke artifact is not mistaken for
or resumed as the 20-epoch run.

### Controlled multi-scale and multi-angle comparisons

The current literature-aligned small baselines keep torchvision's ResNet-18
BasicBlock and replace only the first spatial convolution of every block.  The
second 3x3 convolution, shortcut, stem, stage widths, pooling, classifier, and
training recipe remain unchanged:

| Variant | Spatial extractor before the 1x1 mixer | Pose retained? | Parameters |
| --- | --- | ---: | ---: |
| Standard ResNet-18 | two ordinary 3x3 convolutions | no | 11,227,812 |
| MixConv-4 | 1x1 projection, then channel-split depthwise 3/5/7/9 kernels | no | 7,112,228 |
| Fixed RotInterp-8 | 1x1 projection, one shared depthwise 3x3 bank, eight fixed full-circle bilinear rotations, orientation max | no | 7,050,788 |
| ARC-4bank | 1x1 projection, ARC routing head, four adaptive kernel banks combined before one depthwise convolution | input-adaptive | 7,139,140 |

MixConv-4 follows the paper's mixed-depthwise construction. Fixed RotInterp-8
is an ORN-style reduced baseline: it uses the paper's shared-kernel bilinear
rotation idea but pools the orientation axis inside each replaced layer rather
than carrying an ARF orientation field through the whole network. ARC-4bank follows
the official routing sequence (depthwise 3x3, LayerNorm, ReLU, global average
pooling, sigmoid kernel gates, and softsign angles bounded to +/-40 degrees),
but uses depthwise adaptive kernels so ImageNet-100 batches remain practical.

The earlier whole-block K5/K3 comparison variants remain in the repository for
result provenance, but should be treated as legacy and not used as the paper
baselines.

Five-epoch smoke commands:

```bash
DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
  bash scripts/kaggle/run_imagenet100_resnet18_mixconv4_2xt4.sh

DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
  bash scripts/kaggle/run_imagenet100_resnet18_fixed_rotinterp8_2xt4.sh

DATA_ROOT=/kaggle/input/datasets/ambityga/imagenet100 \
  bash scripts/kaggle/run_imagenet100_resnet18_arc4bank_2xt4.sh
```

Use `EPOCHS=20` only after a variant passes the five-epoch convergence, runtime,
and memory screen.
