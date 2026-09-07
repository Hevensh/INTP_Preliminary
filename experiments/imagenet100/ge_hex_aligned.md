# Hex full6d3r: align the local GE baseline, retain existing subsampling

This is a new from-scratch variant, not a replacement for WV21 or a resume
compatible with its state dict. The ordinary full12d3r moment tokenizer is unaffected.

Changes relative to WV21:
- GroupNorm(1,C), eps=1e-6, over channels and all spatial/orientation positions.
- Content and relative-position scores both scale by 1/sqrt(C).
- Relative PE concatenates separate x/y MLP outputs and a C6 relative-group
  embedding, using the query's inverse rotation for spatial offsets. This aligns
  the local GE implementation's decomposition, not every detail of the paper.
- Linear layers use truncated-normal std 0.02, zero bias. The custom polar bank
  retains its own initialization, as the custom Cartesian lifting bank does in GE.
- Classifier applied at every position/orientation, spatial sum, then per-class
  maximum over orientations (including the spatially summed classifier bias).

Unchanged: full-circle six directions, r3, 144 raw polar prototypes without an
input projection, K24/K12 score sum, widths 144/288/336, depths 2/2/2, three heads,
FFN4, 19-neighbor attention, attention checkpointing and even/even axial selection.
At 224px the stage token counts remain 195/52/14. Geometry, neighbor count, token
count and lifting parameterization still differ from square GE. No strict
equivariance or accuracy improvement is asserted.

Downsampling is deliberately deferred. Even/even axial coordinates already form
a coarser triangular lattice; the unresolved question is the aggregation before
selection. A future option is shared center-plus-six-neighbor pooling followed by
the same selection. Its overlap, boundary normalization and preservation of
orientation channels must be assessed separately; it is not implemented here.

Run (same 20-epoch schedule and batch 256 per GPU):
```bash
!bash scripts/kaggle/run_imagenet100_hex_direction_pyramid_full6r3_ge_aligned_2xt4_e20.sh
```

Validation: explicit attention reference including gradients, GroupNorm layout
equivalence, classifier readout, plus legacy direction-model regression tests.
