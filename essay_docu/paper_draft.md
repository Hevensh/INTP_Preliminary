# SHARE-ViT: Shared Hexagonal Angle-Scale Routing Enhancement for Vision Transformers

> Full first draft — 2026-08-29. The main quantitative table contains only
> completed, schedule-aligned 20-epoch ImageNet-100 runs. Claims that still
> require additional seeds, longer schedules, or downstream tasks are stated as
> limitations rather than conclusions.

## Abstract

Vision Transformers (ViTs) combine global self-attention with a tokenizer that
is usually a square, axis-aligned convolution. This front end offers no explicit
representation of local pose or scale, while absolute positional embeddings
encode where a token is but not which image-conditioned direction it should
attend to. We introduce **SHARE-ViT**, a geometric enhancement that couples a
shared hexagonal angle-scale tokenizer with pose-conditioned attention bias.
Images are sampled on a staggered hexagonal lattice, compared with learned
variable-resolution polar prototypes at two scales and several orientations,
and projected into compact cosine/sine pose coordinates. A null-softmax route
allows a prototype to abstain when no tested pose is suitable. The same matching
principle drives a per-layer, per-head Look field that adds an image-conditioned,
directed bias to self-attention. In controlled 20-epoch training from scratch on
ImageNet-100, a DeiT-Tiny-sized SHARE-ViT with six tested half-turn directions
and compact three-sample-per-radius prototypes reaches 55.04% top-1 and 81.80%
top-5 accuracy, compared with 51.52% and 79.12% for the standard tokenizer. The
gain is obtained with 5.491M parameters versus 5.544M for the baseline. Ablations
show that the geometric tokenizer provides most of the improvement and that
Look bias is complementary to absolute positional embeddings. These results
support explicit local pose and scale as a useful interface between image
sampling and global attention; they do not yet establish exact rotation
equivariance or broad task-level superiority.

## 1. Introduction

Vision Transformers apply self-attention to a sequence of image tokens
[@dosovitskiy2021vit; @touvron2021deit]. Their global interaction is attractive,
but the sequence is normally created by a square patch projection with one
learned weight for each Cartesian location. On a square lattice, axial neighbors
are one grid step away while diagonal neighbors are \(\sqrt{2}\) steps away.
Consequently, equal changes in discrete direction do not correspond to equal
spatial displacements. The tokenizer also entangles appearance with a single
canonical orientation and scale.

Position encoding addresses a different issue. Absolute embeddings tell the
model where a token occurs, and relative position methods tell attention how two
token indices are displaced [@wu2021irpe]. Neither mechanism, by itself, asks
whether the local image evidence supports a particular pose, nor does it turn
that pose evidence into a directed search pattern. Local-window Transformers
such as Swin introduce efficient spatial locality [@liu2021swin], whereas
conditional position encodings generate input-dependent position signals
[@chu2021cpvt]. Our goal is complementary: construct an explicit local
angle-scale representation before global attention and reuse it to condition
where each attention head looks.

Rotation-aware convolution has a long history. Group-equivariant networks share
filters under discrete transformations [@cohen2016gcnn], Active Rotating Filters
materialize oriented responses [@zhou2017orn], and harmonic networks encode
rotation through circular harmonics [@worrall2017harmonic]. Hexagonal
convolution reduces lattice anisotropy and supports additional rotational weight
sharing [@hoogeboom2018hexaconv]. SHARE-ViT borrows the underlying principles of
shared pose transformations and circular coordinates, but it is not a strict
group-equivariant network. Instead, it forms a compact, learnable geometric
front end whose output remains compatible with an ordinary DeiT-Tiny backbone.

Our contributions are:

1. **A shared hexagonal angle-scale tokenizer.** Learned variable-resolution
   polar prototypes are rendered at multiple orientations and scales and matched
   to image patches centered on a common hexagonal lattice.
2. **A compact null-softmax circular-moment projection.** Pose scores are routed
   through an explicit null state and compressed directly with fixed
   cosine/sine coefficients, producing two channels per prototype without a
   learned A/B orthogonal value basis or an independent output vector for every
   pose.
3. **Pose-conditioned Look bias.** Each Transformer layer and head learns a
   directional radial field; image-conditioned pose probabilities rotate and
   rescale this field into a directed additive attention bias.
4. **An aligned ImageNet-100 ablation.** Eight completed 20-epoch runs isolate
   hex sampling, angular resolution, absolute positional embeddings, and Look
   bias under a shared training recipe and closely matched model sizes.

## 2. Related work

### 2.1 Visual tokenization and spatial bias

ViT maps non-overlapping square patches to tokens and adds learned positional
embeddings [@dosovitskiy2021vit]. DeiT demonstrates that a ViT can be trained
effectively on ImageNet without external data using a strong recipe
[@touvron2021deit]. Later models inject more local structure: Swin restricts
attention to shifted windows [@liu2021swin], iRPE introduces directional relative
position terms [@wu2021irpe], and CPVT generates conditional position encodings
from local neighborhoods [@chu2021cpvt]. SHARE-ViT preserves global attention.
Its locality enters through geometric token formation and an input-conditioned
Look field rather than a fixed window or only an index-based lookup table.

### 2.2 Rotation-aware representations

G-CNNs formalize weight sharing over transformation groups
[@cohen2016gcnn]. ORNs rotate active filters and retain orientation channels
[@zhou2017orn], while harmonic networks use circular basis functions to encode
rotation [@worrall2017harmonic]. These methods motivate our shared prototypes and
cosine/sine pose coordinates. The present model, however, samples a finite set of
orientations and trains on ordinary classification images without enforcing an
equivariance identity. We therefore use the terms *pose-aware* and
*rotation-shared*, not *rotation-equivariant*, for the claims evaluated here.

### 2.3 Hexagonal image processing

Hexagonal lattices provide six equidistant first-ring neighbors. HexaConv shows
that hexagonal planar and group convolutions can reduce anisotropy and improve
rotational parameter sharing [@hoogeboom2018hexaconv]. Our input pixels remain
on a conventional Cartesian image. We place patch centers on a staggered
hexagonal lattice and sample circular supports by interpolation; thus the method
does not require native hexagonal camera data or a hexagonal storage format.

## 3. Method

### 3.1 Overview

SHARE-ViT has two coupled paths (Fig. 1). The tokenizer path maps the image to a
sequence of 192-dimensional tokens. The Look path matches image evidence with
separate probes and converts the resulting pose distribution into an additive
bias for every attention layer and head. Both paths share the same geometric
language—scale, direction, polar sampling, and null routing—but do not share
their learned prototype parameters.

![Figure 1. SHARE-ViT architecture.](figs/intp_img_full_architecture.png)

### 3.2 Hexagonal patch centers

Let an image be \(x\in\mathbb{R}^{H\times W\times 3}\). Patch centers are placed
on alternating horizontally shifted rows, yielding a staggered hexagonal token
lattice. For the 224-pixel experiments, the main geometry produces 195 image
tokens, close to the 196 tokens of a standard \(14\times14\) DeiT tokenizer.
Reflection padding is used at image boundaries. Two circular supports with
diameters \(K\in\{24,12\}\) share every center, so scale comparison does not
alter the token graph.

Each support uses a cosine-shaped radial cover \(c_K(u)\). We retain its
accumulated mass rather than dividing by \(\sum_u c_K(u)\). The small-scale
cover is multiplied by

\[
\gamma_K=\frac{\sum_u c_{24}(u)}{\sum_u c_K(u)},
\]

which removes the trivial response difference caused only by the number of
sampled points while preserving the convolution-like response magnitude.

### 3.3 Variable-resolution polar prototypes

The tokenizer contains \(P=96\) learned prototypes. A prototype is stored in
polar coordinates over \(R=12\) radial rings. Ring \(r\) contains
\(q(r+1)\) angular samples, where \(q=4\) for the 4r configuration and \(q=3\)
for the compact 3r configuration. Inner rings therefore do not waste angular
parameters that cannot be resolved near the center, while outer rings retain
more detail. Bilinear polar-to-Cartesian interpolation renders a prototype at a
requested scale and orientation.

For token \(n\), prototype \(p\), scale \(k\), and tested direction \(d\), the
raw matching score is a covered dot product

\[
s_{n p k d}
=\sum_{u\in\Omega_k}
\gamma_k c_k(u)\,
x_{n,k}(u)^\top w_{p,k,d}(u).
\]

No division by \(\lVert x\rVert\lVert w\rVert\) is applied; despite the later
harmonic projection, this is a dot-product kernel rather than cosine
similarity. Scores are summed over the two scales before pose routing:

\[
s_{n p d}=\sum_k s_{n p k d}.
\]

The 4r model evaluates \(D=4\) directions at 45-degree intervals within an
eight-direction global period. The 3r model evaluates \(D=6\) directions at
30-degree intervals within a twelve-direction global period. These half-turn
sets are sufficient for the undirected local structures targeted in the current
experiment while retaining finer phase coordinates in the global period.

### 3.4 Null-softmax and circular-moment projection

A learned scalar \(s^{\varnothing}_p\) is appended to the pose logits for every
prototype. The routing distribution is

\[
[p_{np1},\ldots,p_{npD},p^{\varnothing}_{np}]
=\operatorname{softmax}
([s_{np1},\ldots,s_{npD},s^{\varnothing}_p]).
\]

The null branch contributes no output and the real-pose probabilities are not
renormalized after the null entry is removed. Consequently, a prototype may
reduce its total activation when none of the tested poses matches.

Let \(\theta_d=2\pi d/D_{\mathrm{global}}\). Each prototype emits its first
circular moment,

\[
z_{np}
=\sum_{d=1}^{D}p_{npd}
\begin{bmatrix}\cos\theta_d\\ \sin\theta_d\end{bmatrix}.
\]

Concatenating \(z_{np}\) over 96 prototypes gives a 192-dimensional token. This
factorization shares the output basis across directions and avoids an
independent 192-dimensional value vector for every pose.

### 3.5 Pose-conditioned Look bias

The Look path uses one probe for each of the \(12\times3=36\) layer-head pairs.
Its image sampling is performed without gradient recording, while probe and
field parameters remain learnable. A probe produces null-softmax pose weights
\(\pi_{bq h k d}\) for batch item \(b\), query token \(q\), head \(h\), scale
\(k\), and direction \(d\).

Each head stores one canonical signed field
\(G_h\in\mathbb{R}^{R_L\times D_L}\). In the current models,
\(R_L=2|\mathcal{K}|=4\), while \(D_L\) follows the tokenizer's global
direction period: 8 bins for 4r and 12 bins for 3r. Rotating and radially
resampling \(G_h\) yields a directed token-to-token field
\(F_{hkdqj}\). The bias is

\[
B_{bhqj}=\sum_{k,d}\pi_{bq h k d}F_{hkdqj}.
\]

The null probability has no field and therefore contributes zero. \(B\) is
added to the corresponding attention logits before softmax. Since the field is
conditioned on the query token's local evidence, generally
\(B_{bhqj}\neq B_{bhjq}\).

### 3.6 Transformer backbone

Tokens are consumed by a DeiT-Tiny-sized backbone with embedding dimension 192,
12 blocks, three attention heads, and MLP ratio 4. We compare three positional
settings: learned absolute PE only, Look only, and PE+Look. The classifier and
backbone are otherwise unchanged across the rotating-tokenizer ablations.

## 4. Experiments

### 4.1 Dataset and protocol

We use ImageNet-100 with 100 classes, 130,000 training images, and 5,000
validation images. All reported models are trained **from scratch** for 20
epochs at \(224\times224\) resolution; no pretrained weights or teacher model
is used. Training uses two GPUs with batch size 256 per GPU (global batch 512),
AdamW, initial learning rate \(5\times10^{-4}\), minimum learning rate
\(10^{-6}\), weight decay 0.05, two warmup epochs, cosine decay, label smoothing
0.1, gradient clipping at 1.0, mixed precision, and seed 0.

The training transform is random resized crop with bicubic interpolation,
horizontal flip, RandAugment (2 operations, magnitude 9), ImageNet
normalization, and random erasing with probability 0.25. Validation uses resize
to 256, center crop to 224, and ImageNet normalization. Because only one seed
and a short 20-epoch schedule are currently complete, we treat the results as a
controlled architectural study rather than a definitive benchmark.

### 4.2 Compared configurations

- **Standard:** DeiT-Tiny with its ordinary square \(16\times16\) patch
  projection and absolute PE.
- **Hex only:** a hex-centered patch projection without rotating prototypes.
- **Hex 4r:** 96 variable-ring prototypes, four tested half-turn directions,
  two scales, and null-softmax routing.
- **Hex 3r:** the compact three-samples-per-radius representation with six
  tested half-turn directions and the same two scales.
- **PE only / Look only / PE+Look:** positional-information ablations for each
  rotating tokenizer.

### 4.3 Metrics

We report the best validation top-1 and the top-5 accuracy from the same epoch,
plus total trainable parameter count. Parameter counts are taken from the stored
run summaries. We intentionally omit throughput from the main table: the runs
were executed in shared Kaggle sessions and wall-clock measurements are not yet
controlled tightly enough for a fair systems claim.

## 5. Results

### 5.1 Main ablation

| Tokenizer | Tested directions | Absolute PE | Look bias | Top-1 (%) | \(\Delta\) top-1 | Top-5 (%) | \(\Delta\) top-5 | Params (M) |
|---|---:|:---:|:---:|---:|---:|---:|---:|---:|
| Standard | — | ✓ | — | 51.52 | — | 79.12 | — | 5.544 |
| Hex only | — | ✓ | — | 51.30 | -0.22 | 79.14 | +0.02 | 5.597 |
| Hex 4r | 4 | ✓ | — | 53.56 | +2.04 | 80.40 | +1.28 | 5.486 |
| Hex 4r | 4 | — | ✓ | 50.26 | -1.26 | 77.72 | -1.40 | 5.463 |
| Hex 4r | 4 | ✓ | ✓ | 53.70 | +2.18 | 80.46 | +1.34 | 5.501 |
| Hex 3r | 6 | ✓ | — | 54.54 | +3.02 | 80.46 | +1.34 | 5.464 |
| Hex 3r | 6 | — | ✓ | 52.42 | +0.90 | 79.48 | +0.36 | 5.453 |
| **Hex 3r** | **6** | **✓** | **✓** | **55.04** | **+3.52** | **81.80** | **+2.68** | **5.491** |

The plain Hex tokenizer does not improve top-1 accuracy, indicating that a
staggered lattice alone is insufficient. The rotating geometric tokenizer is
the main source of gain: PE-only Hex 4r and Hex 3r improve top-1 by 2.04 and
3.02 points, respectively. The compact 3r representation with six directions
outperforms 4r with four directions despite using fewer parameters, suggesting
that allocating capacity to additional tested poses is more useful here than
denser angular samples inside each stored ring.

Look-only performance is lower than PE+Look in both direction settings. Thus
the current Look field does not replace absolute location. It is nevertheless
informative: Look-only Hex 3r exceeds the standard PE baseline by 0.90 top-1,
and adding Look to PE improves Hex 3r from 54.54 to 55.04 top-1 and from 80.46
to 81.80 top-5. The larger top-5 change suggests that directed fields reshape
the class ranking beyond the top prediction, although this interpretation
requires confirmation across seeds.

All rotating variants remain close to, or below, the baseline parameter count.
The best model uses 5.491M parameters, 0.053M fewer than Standard. The gain is
therefore not explained by a larger parameter budget.

### 5.2 Qualitative structure

The learned standard patch projection contains unconstrained Cartesian kernels,
whereas the SHARE tokenizers expose learned radial and angular structure
directly. The stored polar prototypes remain compact and can be rendered at
multiple scales and poses. The 3r and 4r visualizations show smooth low-frequency
regions together with localized angular changes, rather than forcing every
radius to use the outer ring's angular resolution.

The Look visualizations show signed, head-specific direction-scale fields. A
positive region raises the corresponding query-key attention logit and a
negative region lowers it. Since each layer-head pair has a separate probe and
canonical field, the learned patterns need not collapse to one global positional
prior. These plots demonstrate that the mechanism learned nonuniform directed
fields; they do not by themselves prove causal reliance or improved
interpretability.

## 6. Discussion

### 6.1 What the ablation establishes

Three conclusions are supported within the aligned setup. First, changing only
the sampling lattice is not enough; learned multi-pose prototype matching is the
important change. Second, absolute PE and Look bias encode complementary
information: the former supplies global token identity, while the latter
supplies image-conditioned directed relations. Third, a compact radial storage
can improve both parameter efficiency and accuracy when its saved capacity is
used for finer pose testing.

### 6.2 What it does not establish

The experiment does not prove exact rotation or scale equivariance. Rendering,
finite direction sets, boundary padding, interpolation, and the unconstrained
Transformer backbone can all break an equivariance identity. We also do not yet
claim ImageNet-1K competitiveness, dense-prediction gains, robustness to
continuous rotations, or hardware efficiency. The current evidence comes from
one seed, one dataset, and a short training schedule.

### 6.3 Next validation

The most important next experiment is an angle-scale controlled benchmark that
separates ordinary classification from pose generalization. Training on one set
of angles/scales and evaluating on interleaved unseen poses would directly test
whether shared prototypes improve interpolation. A second priority is a
multi-seed ImageNet-100 rerun under a longer schedule. Finally, replacing the
first convolution in a parameter-matched residual network would determine
whether the tokenizer's benefit is specific to ViT or general to visual front
ends.

## 7. Conclusion

SHARE-ViT introduces a shared geometric interface between images and global
self-attention. Hex-centered multi-scale patches are matched against rotating
variable-resolution polar prototypes, compressed through null-softmax circular
moments, and reused by a pose-conditioned Look bias. On the current aligned
ImageNet-100 study, the complete six-direction model improves DeiT-Tiny-sized
top-1 accuracy from 51.52% to 55.04% without increasing parameter count. The
result motivates further study of explicit local pose and scale as a complement
to Transformer positional encoding, while leaving exact equivariance,
cross-dataset generalization, and systems efficiency as open questions.

## References

Bibliographic records are maintained in `references.bib`.
