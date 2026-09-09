# Code-to-paper and evidence map

## Method implementation

| Paper component | Source of truth |
|---|---|
| Hex patch centers and image sampling | `layers/hex_patch_geometry.py`, `utils/hex_graph.py` |
| Variable-resolution polar rendering | `layers/polar_prototype.py`, `layers/polar_patch_sampler.py` |
| Main rotating tokenizer | `layers/hex_rotating_harmonic_patch_embed.py` |
| Covered dot-product matching | `layers/rotating_dot_product.py`, Triton kernels under `layers/` |
| Look pose matcher and field renderer | `layers/square_patch_dense_grid_look.py` |
| Token-space Center Look and probe sharing | `layers/center_pose_grid_look.py`, `layers/center_pose_angular_look.py` |
| Directed attention integration | `model/mini_vit.py` |
| DeiT-Tiny-sized SHARE-ViT | `model/deit_tiny_rot_hex_look.py` |
| ImageNet-100 model registry | `experiments/imagenet100/models.py` |
| Dataset discovery and transforms | `experiments/imagenet100/data.py` |
| Reproducible training loop | `experiments/imagenet100/train_vit.py` |

## Main experiment matrix

| Paper label | Stored run directory | Geometric setting | Position setting |
|---|---|---|---|
| Standard | `runs/deit_tiny_imagenet100_ddp_e20` | standard square tokenizer | PE |
| Hex only | `runs/deit_tiny_hex_patch_imagenet100_ddp_e20` | staggered Hex patch projection | PE |
| Hex 4r PE | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_imagenet100_ddp_e20` | half4d4r + null-softmax | PE |
| Hex 4r Look | `runs/deit_tiny_rot_hex_harmonic_softmax_look_imagenet100_ddp_e20` | half4d4r + null-softmax | Look |
| Hex 4r PE+Look | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_imagenet100_ddp_e20` | half4d4r + null-softmax | PE + Look |
| Hex 3r PE | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` | half6d3r + null-softmax | PE |
| Hex 3r Look | `runs/deit_tiny_rot_hex_harmonic_softmax_look_half6_compact_r3_imagenet100_ddp_e20` | half6d3r + null-softmax | Look |
| Hex 3r PE+Look | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` | half6d3r + null-softmax | PE + Look |

## Completed Center Look placement and sharing runs

| Paper label | Stored run directory | Position setting | Kaggle source |
|---|---|---|---|
| Hex 3r Center only | `runs/deit_tiny_rot_hex_harmonic_softmax_center_grid_look_only_share4l_half6_compact_r3_imagenet100_ddp_e20` | G4 Center Look | V24 |
| Hex 3r PE+Image+Center | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` | PE + Image Look + G4 Center Look | V23 |
| Hex 3r PE+Image+Center G1 | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share1l_half6_compact_r3_optimized_imagenet100_ddp_e20` | PE + Image Look + G1 Center Look | XV1-R1 |
| Hex 3r PE+Image+Center G2 | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share2l_half6_compact_r3_optimized_imagenet100_ddp_e20` | PE + Image Look + G2 Center Look | XV1-R2 |
| Hex 3r PE+Image+Center G3 | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share3l_half6_compact_r3_optimized_imagenet100_ddp_e20` | PE + Image Look + G3 Center Look | XV2-R1 |
| Hex 3r PE+Image+Center G6 | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share6l_half6_compact_r3_optimized_imagenet100_ddp_e20` | PE + Image Look + G6 Center Look | XV2-R2 |
| Hex 3r PE+Image+Center G12 | `runs/deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share12l_half6_compact_r3_optimized_imagenet100_ddp_e20` | PE + Image Look + G12 Center Look | XV2-R3 |

## Paper artifacts

| Artifact | Purpose |
|---|---|
| `essay_docu/paper_draft.md` | complete first manuscript draft |
| `essay_docu/main.tex` / `main.pdf` | compiled two-column manuscript and rendered draft |
| `essay_docu/claim_ledger.md` | evidence boundaries and prohibited overclaims |
| `essay_docu/references.bib` | primary literature records |
| `essay_docu/imagenet100_core_results_table.tex` | publication-style main table source |
| `essay_docu/imagenet100_core_results_table.pdf` | rendered main table |
| `essay_docu/figs/01_share_vit_full_architecture.pdf` | full SHARE-ViT architecture diagram |
| `essay_docu/figs/02_hex_multiscale_multiangle_kernels.pdf` | multi-scale/multi-angle tokenizer illustration |
| `essay_docu/figs/03_half6d3r_look_bias_fields.pdf` | 3r Look fields |
| `essay_docu/figs/08_all5000_six_models_rotation_fans.pdf` | Six-model full-validation local-rotation comparison |

The LaTeX main text references the approved local copies under
`essay_docu/figs`, so the paper workspace is self-contained. Editable figure
sources and assembly intermediates remain under `essay_docu/figure_sources`.

## Excluded from the main paper table

- Failed or interrupted Kaggle runs.
- Five-epoch smoke tests once a corresponding aligned 20-epoch run exists.
- Earlier dense Look implementations and obsolete tokenizer branches.
- CIFAR diagnostics, except as explicitly labeled secondary analysis.
- Remote-sensing concepts until completed artifacts exist.
