# SHARE-ViT

**SHARE-ViT** is a **S**hared **H**exagonal **A**ngle-Scale **R**outing
**E**nhancement for Vision Transformers. It augments a conventional ViT with
an explicitly local geometric interface while preserving the Transformer's
global self-attention backbone.

The repository contains the research implementation, diagnostics, ablations,
and reproducible ImageNet-100 training entry points used to study this design.

## Motivation

A standard ViT divides an image into a square lattice and projects every patch
with a large-stride convolution. This is simple and efficient, but the induced
local neighborhood is directionally uneven: horizontal and vertical patch
centers are one grid step apart, whereas diagonal centers are `sqrt(2)` grid
steps apart. Ordinary patch projection also does not explicitly share a local
pattern across changes in orientation and scale.

CNNs provide strong locality, while Transformers provide long-range context.
SHARE-ViT connects these strengths by:

1. arranging patch centers on a Hex lattice with six equidistant first-ring
   neighbors;
2. matching shared compact local geometry across discrete angles and scales;
3. routing pose evidence through a null-aware distribution instead of forcing
   every patch into an explicit pose; and
4. optionally converting the same pose evidence into a directed Look Bias for
   each Transformer layer and attention head.

The intended contribution is **pose-aware local geometric sharing**, not a
claim of strict rotation equivariance, universal rotation invariance, or lower
compute in the current research implementation.

## Method overview

### 1. Hex patch sampling

`HexPatchGeometry` extracts overlapping local supports whose centers form a Hex
lattice. Multiple kernel sizes share exactly the same centers, so their
responses can be combined without changing the token graph. The current
ImageNet-100 mainline uses two supports, `24x24` and `12x12`, with approximately
the same token count as DeiT-Tiny at `224x224` input resolution.

### 2. Shared angle-scale routing

The tokenizer stores compact variable-resolution polar geometry rather than a
separate dense kernel for every pose. A sampler renders the same stored geometry
at several orientations and spatial supports. Dot-product responses are routed
with a softmax that includes a learned null state, allowing weakly matched
patches to abstain.

The routed directional probabilities are projected through cosine and sine
circular moments. With 96 geometric channels, the paired projections form the
192-dimensional token expected by DeiT-Tiny.

### 3. Pose-conditioned Look Bias

The Look module maps local pose evidence to a signed direction-by-radius field.
Each Transformer layer and attention head owns its own learned field. The field
is added to attention logits, providing a directed preference for where a token
should look while standard self-attention continues to model global relations.

Look direction resolution follows the tokenizer's full orientation period; its
radial resolution is twice the number of tokenizer scales. For example, the
current `half6d3r` tokenizer uses a `12 direction x 4 radius` Look field.

An experimental deep-feature refinement keeps the original image-derived Look
path and augments all twelve Transformer blocks by default. For every query it
reads only the first Hex graph ring and correlates its six neighbors over C6.
Each layer/head stores six shared `head_dim -> 1` projections in polar
`(radius, phase)` form, so rotating a candidate synchronizes spatial shifts and
feature-pair rotation. The resulting local pose evidence rotates one learned
canonical `4 radius x 12 direction` map: a cheap one-ring detector therefore
decides which parts of the full four-ring attention field should be emphasized.
No C12 detector, second feature scale, or short-ring FFT is evaluated.

## Main experiment configurations

- `standard`: ordinary DeiT-Tiny patch projection.
- `Hex`: Hex sampling without explicit orientation routing.
- `half4d4r`: four sampled poses over an eight-direction full period, with four
  angular samples per successive polar radius.
- `half6d3r`: six sampled poses over a twelve-direction full period, with three
  angular samples per successive polar radius.
- `PE only`, `Look only`, and `PE + Look`: controlled positional-information
  ablations.

The configuration names describe discrete research settings; they do not imply
continuous group equivariance.

## ImageNet-100 results

All values below come from completed 20-epoch runs with `224x224` inputs on the
same 100-class ImageNet-100 dataset and a two-T4 training environment.

| Tokenizer | Directions | Position information | Top-1 | Top-5 |
|---|---:|---|---:|---:|
| Standard DeiT-Tiny | - | PE | 51.52 | 79.12 |
| Hex | - | PE | 51.30 | 79.14 |
| SHARE `half4d4r` | 4 | PE | 53.56 | 80.40 |
| SHARE `half4d4r` | 4 | Look | 50.26 | 77.72 |
| SHARE `half4d4r` | 4 | PE + Look | 53.70 | 80.46 |
| SHARE `half6d3r` | 6 | PE | 54.54 | 80.46 |
| SHARE `half6d3r` | 6 | Look | 52.42 | 79.48 |
| **SHARE `half6d3r`** | **6** | **PE + Look** | **55.04** | **81.80** |

These experiments support the claim that the tested SHARE-ViT configurations
improve over the aligned standard baseline under this training budget. They do
not establish state of the art or universal superiority across datasets,
backbones, and schedules.

## Repository map

- [`layers/`](layers): Hex geometry, rotating tokenizers, matching kernels, and
  Look Bias implementations.
- [`model/`](model): model integrations and research architectures.
- [`experiments/imagenet100/`](experiments/imagenet100): ImageNet-100 models,
  training entry points, and experiment documentation.
- [`configs/`](configs): externalized experiment definitions.
- [`diagnostics/`](diagnostics): analysis pipelines and interactive
  visualization service.
- [`scripts/kaggle/`](scripts/kaggle): Kaggle setup and multi-GPU launch scripts.
- [`scripts/figures/`](scripts/figures): reproducible static paper and supporting
  material figures.
- [`essay_docu/`](essay_docu): manuscript draft, claim ledger, and experiment map.
- [`tests/`](tests): geometry, routing, model, and kernel equivalence tests.

## Reproduction

The ImageNet-100 workflow expects the Kaggle dataset
[`ambityga/imagenet100`](https://www.kaggle.com/datasets/ambityga/imagenet100).
Dataset discovery, launch commands, output structure, and checkpoint handling
are documented in
[`experiments/imagenet100/README.md`](experiments/imagenet100/README.md).

Generated datasets, checkpoints, diagnostic arrays, downloaded papers, and
training outputs are intentionally excluded from source control. The `runs/`
index records external experiment provenance without committing large weights.

## Project status

SHARE-ViT is exploratory research software. The current evidence is strongest
for image classification ablations on ImageNet-100. Detection, segmentation,
larger training budgets, continuous-angle evaluation, and optimized inference
remain future validation targets.

Earlier files and checkpoints may retain the development name `INTP` or
`INTP-Img`. Those names are preserved only for experiment traceability; new
documentation and figures use **SHARE-ViT**.

## License

Source code is released under the [MIT License](LICENSE). Third-party datasets,
papers, pretrained weights, and referenced assets remain subject to their own
licenses.
