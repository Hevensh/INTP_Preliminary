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

## Optional retained-center pooling experiment

The separate `hex_direction_pyramid_full6_ge_aligned_pool7` variant now implements
the proposed center-plus-six-neighbor max pooling. It gathers seven positions
only for retained coarse centers, then pools independently per direction/channel
and applies the existing shared Linear. Missing boundary neighbors are masked
with negative infinity. No dense fine-grid pooled tensor is constructed.
Token counts and trainable parameter count (5,499,456) are unchanged. The original
aligned selection-only configuration remains unchanged for comparison.

```bash
!bash scripts/kaggle/run_imagenet100_hex_direction_pyramid_full6r3_ge_aligned_pool7_2xt4_e20.sh
```

Validated: 11 tests including a direct neighborhood pooling/gradient reference,
and 224px batch-2 CUDA AMP forward/backward. Full training speed is not measured.

## Memory-bounded aligned attention

Aligned variants now gather at most 16 query neighborhoods at once, checkpoint
each gather/attention chunk separately, and checkpoint norm2/FFN. QKV remains
shared across chunks. This avoids retaining all expanded neighborhoods during
backward recomputation of a whole-attention checkpoint. It changes neither
parameters nor pooling, attention windows, batch size, or optimization schedule.
The same pool7 script selects this implementation; old checkpoints remain loadable.

Local CUDA AMP AdamW, batch16, 224px, three steps: peak allocated memory fell
from 851.13 MiB to 437.50 MiB (48.6%). Warm steps were about 87-88 ms before,
109-124 ms after. These are local microbenchmarks, not T4 batch256 guarantees.
Twelve tests pass including full/chunked attention input and parameter gradients.
