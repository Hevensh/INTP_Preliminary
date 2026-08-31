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
No C12 detector or second feature scale is used. To avoid repeating the same
neighbor gather in every block, layers are divided into stages of four. The
stage-entry hidden state generates four independent, layer-specific Look fields
in one batched operation; the later three blocks therefore use routing evidence
from the start of their stage. A mathematically equivalent C6 frequency-domain
path is retained as an ablation, but direct batched correlation is the default:
on the local RTX 4060 training-step benchmark it was slightly faster and used
less peak memory than the short-ring FFT path.

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

The experimental ResNet branch also includes a stage-routed replacement. It
removes the first `3x3` convolution from every ResNet-18 BasicBlock and computes
one coarse, large-support MAMS route at each stage entrance. Same-resolution
blocks use their input as the residual-branch seed; transition blocks reuse the
existing shortcut projection, so no replacement input-side `1x1` is added. The
two blocks in a stage share the costly pose/scale matching probabilities but
retain independent low-rank A/B/Vscale values.
Routes are generated at `14x14, 14x14, 14x14, 7x7` and interpolated onto the
stage feature grid; this avoids applying a large dynamic kernel densely at all
early-stage positions. The four stage supports are respectively `9/5`, `9/5`,
`7/3`, and `5/3`.

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

### Center Look probe-sharing ablation

The Center Grid Look branch evaluates a direction probe from tokenizer features
and routes it through an independent learned `4 radius x 12 direction` map for
every Transformer layer and head.  `G` is the number of consecutive layers that
reuse one probe evaluation: a new probe is evaluated at layers `0, G, 2G, ...`,
and the final group may contain fewer than `G` layers.  The output Look maps are
never shared.  Center Look affects the first eleven blocks because a
patch-to-patch bias in the final block cannot affect that block's CLS output.

| Kaggle run | Layers per probe (`G`) | Probe groups | Best Top-1 | Top-5 at best Top-1 | Parameters | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| V19-R1 | 1 | 11 | 54.64 | 81.86 | 5.478M | 146.7 min |
| V19-R2 | 2 | 6 | 54.64 | 81.56 | 5.472M | 146.7 min |
| V20-R1 | 3 | 4 | 55.00 | 81.56 | 5.470M | 155.8 min |
| **V20-R2** | **4** | **3** | **55.18** | **81.66** | **5.469M** | **155.1 min** |
| V20-R3 | 6 | 2 | 54.46 | 81.60 | 5.467M | 155.0 min |

All five rows use `half6d3r`, tokenizer null-softmax, PE, and Center Grid Look;
only `G` changes.  G4 is the current best setting, but its 0.18-point Top-1
margin over G3 is small.  G6 loses accuracy, suggesting that moderate local
sharing is preferable to making the direction probe too depth-invariant.  The
near-identical runtimes within each Kaggle version also show that probe
evaluation is not the dominant cost; the V19/V20 runtime offset should be
treated as environment variation rather than an architectural speed result.
G12, which evaluates one probe for the final incomplete eleven-layer group, is
scheduled as the global-sharing endpoint.

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
