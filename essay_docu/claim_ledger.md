# Claim ledger

This ledger separates implementation facts, completed measurements, bounded
interpretations, and unsupported claims. The paper may summarize a row only
within its stated boundary.

## Implementation claims

| Claim | Evidence location | Status | Boundary |
|---|---|---|---|
| Patch centers form a staggered hexagonal lattice and both scales share centers | `layers/hex_patch_geometry.py`, `layers/hex_rotating_harmonic_patch_embed.py` | code-validated | implementation fact |
| Main tokenizer uses K24/K12 circular supports with cover-mass compensation | `layers/hex_rotating_harmonic_patch_embed.py` | code-validated | no claim that this is optimal |
| 96 prototypes emit paired cos/sin coordinates to form a 192-D token | `layers/hex_rotating_harmonic_patch_embed.py` | code-validated | tokenizer mainline only |
| Matching is a covered dot product, not cosine similarity | `layers/rotating_dot_product.py` | code-validated | current dot-product mainline |
| Null route contributes zero and real-pose probabilities are not renormalized | tokenizer and `layers/square_patch_dense_grid_look.py` | code-validated | current null-softmax mainline |
| Look fields are independent for 12 layers × 3 heads | `model/deit_tiny_rot_hex_look.py` | code-validated | DeiT-Tiny-sized model |
| Look grid has four radial bins and direction bins equal to the tokenizer global period | `model/deit_tiny_rot_hex_look.py` | code-validated | 8×4 for 4r, 12×4 for 3r |
| Look bias is directed and image-conditioned | `layers/square_patch_dense_grid_look.py` | code-validated | architectural property, not a causal result |
| Center Look reads bias-corrected tokenizer cos/sin pairs, split as 32 pairs per head | `layers/center_pose_grid_look.py`, `model/deit_tiny_rot_hex_look.py` | code-validated | tokenizer-space pose probe, not intermediate Transformer features |
| Center Look shares pose probes while preserving per-layer/head 4x12 fields | `layers/center_pose_grid_look.py` | code-validated | first 11 blocks only |

## Completed 20-epoch ImageNet-100 measurements

| Configuration | Top-1 | Top-5 | Params | Evidence | Status |
|---|---:|---:|---:|---|---|
| Standard + PE | 51.52% | 79.12% | 5.544M | `runs/deit_tiny_imagenet100_ddp_e20` | validated artifact |
| Hex only + PE | 51.30% | 79.14% | 5.597M | `runs/deit_tiny_hex_patch_imagenet100_ddp_e20` | validated artifact |
| Hex 4r + PE | 53.56% | 80.40% | 5.486M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_imagenet100_ddp_e20` | validated artifact |
| Hex 4r + Look | 50.26% | 77.72% | 5.463M | `runs/deit_tiny_rot_hex_harmonic_softmax_look_imagenet100_ddp_e20` | validated artifact |
| Hex 4r + PE + Look | 53.70% | 80.46% | 5.501M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_imagenet100_ddp_e20` | validated artifact |
| Hex 3r + PE | 54.54% | 80.46% | 5.464M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| Hex 3r + Look | 52.42% | 79.48% | 5.453M | `runs/deit_tiny_rot_hex_harmonic_softmax_look_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| Hex 3r + PE + Look | 55.04% | 81.80% | 5.491M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |

All eight rows use ImageNet-100, 224×224 input, training from scratch, 20
epochs, global batch 512, AdamW, and the aligned augmentation/schedule described
in `paper_draft.md`. Values are the best validation top-1 and the top-5 from the
same epoch. Only one seed is complete.

## Full-validation local-rotation comparison

| Model | 0° Top-1 | 0° Top-5 | Mean local Top-1 | Mean local Top-5 | Mean JSD | Params |
|---|---:|---:|---:|---:|---:|---:|
| Standard DeiT-Tiny | 51.54% | 79.12% | 49.91% | 77.45% | 0.0527 | 5.544M |
| SHARE half4d4r + Image Look | 53.68% | 80.46% | 51.25% | 78.89% | 0.0565 | 5.501M |
| SHARE half6d3r + dual Look | 55.52% | 81.80% | 52.82% | 79.98% | 0.0564 | 5.497M |
| Equi-ViT / GMR | 41.52% | 70.40% | 41.08% | 70.12% | 0.0322 | 5.424M |
| ARC Adaptive | 49.38% | 76.56% | 47.76% | 75.70% | 0.0525 | 5.986M |
| GE-ViT p4 local | 56.06% | 82.04% | 52.94% | 79.45% | 0.0588 | 5.509M |

These values come from one shared local evaluation stream over all 5,000
validation images. The local means exclude 0° and average the twelve rotations
from -30° to +30° at 5° increments. JSD is measured against each model's own
0° probability vector and is confidence-sensitive; it is not a standalone
robustness ranking. Aggregate evidence is stored in
`runs/rotation_consistency/all5000_six_models_local5.json`, with paired
per-image outputs in the adjacent `.samples.npz` archive.

## Completed Center Look sharing measurements

| Layers per pose probe G | Probe groups | Top-1 | Top-5 | Params | Status |
|---:|---:|---:|---:|---:|---|
| 1 | 11 | 54.64% | 81.86% | 5.478M | validated artifact |
| 2 | 6 | 54.64% | 81.56% | 5.472M | validated artifact |
| 3 | 4 | 55.00% | 81.56% | 5.470M | validated artifact |
| 4 | 3 | 55.18% | 81.66% | 5.469M | validated artifact |
| 6 | 2 | 54.46% | 81.60% | 5.467M | validated artifact |
| 12 | 1 | 54.48% | 81.38% | 5.466M | validated artifact |

All rows use Hex half6d3r + null-softmax + PE and differ in pose-probe sharing
span. Layer output fields remain independent. Only one seed is complete, so the
small differences are not statistically conclusive.

## Completed Center Look placement and interaction measurements

| Position setting | Top-1 | Top-5 | Params | Stored run | Status |
|---|---:|---:|---:|---|---|
| PE only | 54.54% | 80.46% | 5.464M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| Image Look only | 52.42% | 79.48% | 5.453M | `runs/deit_tiny_rot_hex_harmonic_softmax_look_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| PE + Image Look | 55.04% | 81.80% | 5.491M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| PE + G3 Center Look | 55.00% | 81.56% | 5.470M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share3l_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| PE + Image Look + G3 Center Look | 55.54% | 81.82% | 5.497M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share3l_half6_compact_r3_optimized_imagenet100_ddp_e20` | validated artifact |
| PE + G4 Center Look | 55.18% | 81.66% | 5.469M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| G4 Center Look only | 52.34% | 79.22% | 5.431M | `runs/deit_tiny_rot_hex_harmonic_softmax_center_grid_look_only_share4l_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |
| PE + Image Look + G4 Center Look | 54.96% | 81.84% | 5.496M | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` | validated artifact |

All rows use the half6d3r tokenizer and the same 20-epoch schedule. At G3, the
two Look branches improve top-1 by 0.50 points over PE + Image Look and by 0.54
points over PE + G3 Center Look. The older G4 three-way run does not show this
gain, so the supported conclusion is a placement-by-sharing-span interaction,
not universal redundancy or unconditional additivity. The G4 three-way run
predates the optimized structured Look kernel and should be rerun before making
a strict G3-versus-G4 claim.

## Supported interpretations

| Interpretation | Status | Boundary |
|---|---|---|
| Best 3r PE+Look improves Standard by +3.52 top-1 and +2.68 top-5 | supported | aligned 20-epoch ImageNet-100 setup only |
| Gain is not due to more parameters | supported | best model has 5.491M vs 5.544M baseline |
| Rotating matching matters more than hex sampling alone | supported | Hex only is -0.22 top-1; rotating PE variants are positive |
| Absolute PE and Look are complementary | supported | PE+Look exceeds Look-only in both 4r and 3r; Look adds +0.50 top-1 to 3r PE |
| PE and G4 Center Look are complementary in the tested placement ablation | supported | 55.18% vs 54.54% PE only; one seed only |
| Image Look and Center Look are complementary at G3 | supported | 55.54% vs 55.04% PE+Image Look and 55.00% PE+G3 Center Look; one seed only |
| Dual-Look gain is independent of Center Look sharing span | not supported | G3 reaches 55.54%, while available G1/G2/G6/G12 results are lower and legacy G4 reaches 54.96% |
| 3r/six-direction allocation is better than 4r/four-direction allocation | supported | under these two tested configurations only; storage density and direction count change together |
| Larger Look top-5 gain reflects changed class ranking | plausible | descriptive interpretation; needs multi-seed confirmation |

## Secondary diagnostics

| Claim | Evidence | Status | Boundary |
|---|---|---|---|
| Tokenizer can approximate a DeiT patch embedding with mean token cosine 0.9860 | CIFAR tokenizer distillation artifact | validated setup | not a superiority claim |
| Recorded frozen-transfer accuracy reaches 77.1% | CIFAR transfer artifact | validated setup | not aligned with ImageNet main table |
| Compact Look path was faster/lower-memory than an earlier dense path locally | local compact benchmark | locally validated | hardware/version dependent; omit from main result table |

## Claims prohibited by current evidence

- Exact rotation or scale equivariance.
- Universal superiority over ViT, DeiT, CNNs, or equivariant networks.
- ImageNet-1K, detection, segmentation, or remote-sensing gains.
- A hardware-efficiency advantage based on the shared Kaggle runs.
- Statistical significance across random seeds.
