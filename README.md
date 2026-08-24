# INTPcore

Experimental image models built around geometric prototypes, multi-angle and
multi-scale matching, Hex tokenization, and pose-conditioned look bias.

The repository currently contains research code and diagnostics for:

- mixed geometric prototype tokenizers;
- polar, angular, stripe, color, and full prototype families;
- DeiT-Tiny / Hex-ViT transfer experiments;
- dense directional look-bias experiments;
- synthetic rotation-and-scale classification controls;
- reproducible ImageNet-100 experiments on Kaggle.

## ImageNet-100 baseline

The first public 224 px benchmark is a five-epoch, randomly initialized
DeiT-Tiny smoke test on
[`ambityga/imagenet100`](https://www.kaggle.com/datasets/ambityga/imagenet100).
See [`experiments/imagenet100/README.md`](experiments/imagenet100/README.md) for
the Kaggle commands and saved artifacts.

The smoke run measures convergence, runtime, throughput, and memory before a
full comparison. Later configurations will compare the same baseline with
Hex-INTP + look bias and a multi-stage INTP network.

## Status

This is exploratory research software. Interfaces and experiment definitions
may change as ablations are consolidated. Generated datasets, model
checkpoints, diagnostic arrays, and training outputs are intentionally excluded
from source control.

## License

Source code is released under the [MIT License](LICENSE). Third-party datasets,
papers, pretrained weights, and other referenced assets remain subject to their
respective licenses.
