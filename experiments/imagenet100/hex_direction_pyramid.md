# Hex direction pyramid: GE-sized widths, polar tokenizer

Current: widths144/288/336, heads3/3/3, depths2/2/2, FFN ratio4.
Hidden widths576/1152/1344. Parameters5,510,244 versus GE5,508,616 (0.030%
difference). Replaces the untrained FFN8.25 setup; run name is distinct.

First layer remains OUR144 polar r3 prototypes (3x234 values each), six poses
0/30/60/90/120/150 degrees, K24/K12. Raw two-scale summed responses:
B,N,6,144. No tokenizer softmax/null/cos-sin projection, no Cartesian kernel swap.

Token counts195/52/14; two-ring19-position attention at every stage. Even/even
axial selection after each first two stages, followed by shared linear projection.
All queries per stage processed together; attention-only checkpointing in the
first two stages avoids full FFN recomputation.

This matches GE widths/depths/FFN, NOT every operation: channel LayerNorm, mean
readout, initialization and Hex selection still differ. Half-circle directions
are not a closed group; no strict C6 equivariance claim. Equal parameters are
not equal FLOPs.

Local RTX4060 synthetic224px batch16 AMP+AdamW: ~0.067s/step after warmup,
893MB peak allocated, versus ~0.29s for the old flat model in the same invocation.
No data loading/DDP. Not a T4 epoch estimate; batch256/GPU untested on T4.
Geometry tests and full model backward pass.

Run20epochs (20-epoch LR horizon,2warmup, LR5e-4, seed0, batch256/GPU,2GPUs):

`!bash scripts/kaggle/run_imagenet100_hex_direction_pyramid_2xt4_e20.sh`

No full training launched locally. Historical code is retained in Git.
