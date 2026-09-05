# Sparse Hex dual Look experiment

Run on Kaggle (after clone), with the final argument specifying Feature probe-sharing span G:

```bash
!bash scripts/kaggle/run_imagenet100_sparse_hex_dual_look_2xt4_e20.sh 3
```

- 20 epochs, two GPUs, 256 images/GPU, unchanged LR/seed/data defaults.
- Image M4, Feature M4 with independent W for each of six directions.
- Differentiation at epochs 3/5/7; full fractions 3/4, 1/2, 1/4.
- Centers form a translated axial sublattice with basis (2,2), (-2,4), anchored at the token nearest the grid mean. Current 224 input has 195 tokens and 17 Look centers.
- Each center addresses inner6 + outer12 neighbors. Adjacent domains share exactly three outer-edge tokens. Finite-image missing neighbors are discarded; no wrapping or mass renormalization.
- K12 probabilities control inner6; K24 controls outer12. Each probe stores 18 coefficients. Feature probabilities control all18. Probe outputs are averaged; both branches retain null-softmax with zero null output.
- Ring-local permutations approximate the six half-angle poses: inner ring uses round-half-up index shifts; outer ring uses one-position shifts. They are bijections per ring, not exact 30-degree spatial rotations.
- Matching is performed only at selected query centers. Tokenizer remains dense. All queries retain global key/value attention. Only selected rows receive Look bias; the others use ordinary SDPA. Selected rows use explicit FP32 attention, avoiding a full N-by-N bias field and avoiding Look interpolation.
- New experiment directory contains `indW_sparsehex18`, so old M4 checkpoints are not resumed. G changes sharing only, not center spacing.

Validation: 15 tests (including dense-reference value/gradient equivalence, geometry, ring-scale aggregation, model roundtrip and G1/3/12 backward). Local RTX4060 Laptop, batch8, 224, FP16 AMP, five measured steps after three warmups: dense independent M4 median 75.80ms, sparse median 64.54ms. This short synthetic check is not a T4 epoch-time prediction.

`summary.json` experiment diagnostics record query indices, axial coordinates, valid neighbor mappings, approximate rotation permutations, G, probe counts, and learned templates.
