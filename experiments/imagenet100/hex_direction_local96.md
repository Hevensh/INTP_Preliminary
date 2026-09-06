# Explicit six-direction Hex local-attention prototype

This is a new experimental architecture, not a GE-ViT reproduction and not a
compute-matched comparison. Existing SHARE and GE variants are unchanged.

## Configuration

- 224px input, Hex stride18, 96 compact r3 prototypes, K24/K12.
- Six raw dot responses at 0/30/60/90/120/150 degrees, stored as B,N,6,96.
- Scale scores are summed, retaining the existing cosine-cover mass compensation.
- No tokenizer null, softmax or cos/sin output projection. No absolute PE or Look.
- Twelve constant-width96 pre-LayerNorm blocks; three heads, FFN ratio4.
- Shared QKV and FFN across directions. Exact axial two-ring neighbors: 19
  positions including self, times six key directions =114 candidates/query.
- Content score plus query-dependent relative PE. Spatial coordinates are
  expressed in the query frame; a small MLP encodes them. Another MLP encodes
  cos/sin of the actual key-minus-query angle (not modulo six slot indices).
- Attention softmax uses the ordinary head-dimension scaling. Boundary positions
  are masked; no wrapping. Output keeps six directions through all blocks.
- Mean over tokens and directions, then classification.
- Query chunks of16 and non-reentrant block checkpointing bound training memory.

The half-circle slots are not closed under60-degree rotations for arbitrary
directed kernels. The two proposed three-slot subsets are NOT treated as exact
cyclic groups. Polar kernel rendering interpolates sampled kernels; there is no
additional fabricated intermediate feature slot or rounded spatial rotation.
This prototype makes no exact C6-equivariance claim.

## Smoke test

`!bash scripts/kaggle/run_imagenet100_hex_direction_local96_2xt4_e5.sh`

Two GPUs, batch256/GPU, same data/augmentation defaults and seed0; five epochs
with two warmup epochs and a twenty-epoch cosine schedule. The five-epoch run
stops early without compressing the learning-rate schedule. Future smoke configs
must set `schedule_epochs` to the intended full training horizon; `epochs` only
controls stopping. Keep batches, sample count and optimizer steps per epoch
matched when comparing early curves. Historical configs are not rewritten.
No full training has been performed locally.

Full run: `!bash scripts/kaggle/run_imagenet100_hex_direction_local96_2xt4_e20.sh`.

1,433,668 parameters. CPU geometry/chunk/gradient/model tests pass. Local RTX4060
224px AMP forward/backward passes. A synthetic batch16 with AdamW measured
about0.33s/step after initial warmup and397MB peak allocated memory. This excludes
data loading and DDP, and is NOT a T4 epoch-time estimate. Fair comparisons must
measure the actual stage widths, FLOPs and throughput, not just direction count.
