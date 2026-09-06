# half7d5r Image Look: fixed 4x12 field

This 20-epoch experiment changes only the Image Look output template relative
to `deit_tiny_rot_hex_harmonic_softmax_pe_look_half7_compact_r5_ddp_e20.json`
(apart from its run name and equivalent sparse interpolation implementation).

- Tokenizer and Image Look probe: seven poses from a fourteen-direction period,
  compact r5 polar storage, K24/K12; pose angles remain spaced by 360/14 degrees.
- Image Look field: four rings, twelve angular bins per ring (48 weights), not
  the previous 5/10/15/20-bin rings (50 weights). Field bins are interpolation
  coordinates, not additional probe directions.
- PE, null-softmax, one Image Look probe per head/layer, no Feature Look,
  no differentiation; batch 256 per GPU, two GPUs; other training settings unchanged.
- Fixed geometry interpolation uses a precomputed sparse matrix and its transpose
  in backward. Tests compare values and gradients against the regular bilinear renderer.
- New run directory avoids resuming a variable-field checkpoint.

The reported 54.6% variable-field result motivates this ablation, but does not
establish causality. The variable field allocates fewer angular bins near the
query (5 versus 12), and more far away (20 versus 12). This changes spatial
resolution and interpolation smoothing, not just parameter count. Restoring the
older field does not guarantee restoring its accuracy with a different tokenizer.

Local validation: CPU/CUDA interpolation value and gradient tests; 224x224,
batch-two AMP full-model forward/backward with finite gradients. Parameter count:
5,552,368. No full training or T4 throughput claim is made by these checks.
