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
