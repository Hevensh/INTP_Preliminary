# Portable ImageNet-100 path index

`ambityga_imagenet100.json.gz` contains filenames and class labels only, not
images or model weights. Source: the real index exported by xiongwutao's Kaggle
`intp-img-littletest`, version 18, for `ambityga/imagenet100`.

The loader tries the session cache first, then this bundled index. Relative
paths support different mount locations. Split names, class order, labels and
first/last files per split/class are checked. It does not verify all file
contents or detect every dataset revision. For a modified dataset, set
`INTP_DISABLE_BUNDLED_INDEX=1` to force normal indexing (and use a fresh session
cache directory). Missing or mismatched manifests fall back to normal indexing.

To regenerate from an actual session cache:

```bash
python scripts/kaggle/export_portable_index.py CACHE.json experiments/imagenet100/manifests/ambityga_imagenet100.json.gz
```

Preserving sample order is important: sorting the list again changes seeded
training permutations. The exporter preserves the original cache order.
