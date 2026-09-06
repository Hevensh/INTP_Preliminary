# Six-direction Hex pyramid: 96 / 192 / 288

Independent variant `hex_direction_pyramid`; old experiments remain available.
Each stage has two two-ring local attention blocks, including the central token.
Stage widths are96/192/288, heads3/3/3, token counts195/52/14 for224px input.
All stages use three heads, with per-head dimensions32/64/96.
FFN ratio8.25 gives hidden widths792/1584/2376. Total parameters5,494,660,
versus5,508,616 for the local GE baseline (0.25% fewer). This matches parameter
count, NOT FLOPs, capacity allocation or the exact GE architecture. Optimizer,
learning rate5e-4,20-epoch schedule and batch256/GPU remain unchanged.

After each of the first two stages, select even/even axial coordinates relative
to the first token and apply a direction-shared linear projection. Coordinates
are divided by2 for the next neighborhood construction. This is selection after
learned neighborhood aggregation, not GE's square max-pool, and not a proven
anti-aliasing filter. Boundary neighbors are masked. Six half-circle slots remain
non-closed under rotation: no strict equivariance claim.

The tokenizer remains raw two-scale r3 dot matching,96 prototypes, six poses.
No new initialization, readout, normalization or PE ablation is bundled here.
Mean readout and channel LayerNorm remain as in the prior96-wide prototype.

## Speed and memory

All queries of a stage are processed in one attention call instead of16-query
chunks. There are six attention calls per forward, versus156 previously.
Checkpoint only attention in stages1/2, not the whole FFN/block; stage3 needs no
checkpoint. SDPA is used but no specific Flash backend is guaranteed on T4.

Historical 96/192/256 version ONLY: local RTX4060 synthetic224px training
steps (AMP, AdamW; no data loading or DDP). Not current288-width measurements:

| Batch | Old12-layer model | Pyramid | Old/pyramid peak allocated |
|---|---:|---:|---:|
|16|~0.37s|~0.046s|398/661MB|
|64|~0.76s|~0.174s|1461/2430MB|

Measurements follow warmup in the same process. Faster at the cost of more
temporary memory. Batch256/GPU remains the training default, but has NOT been
validated on T4; batch64 memory is not a guaranteed linear extrapolation.
Historical256-width parameter count:2,866,116. Eight CPU/CUDA tests pass, including checkpoint and
query chunk gradient equivalence; full224px AMP backward/optimizer steps pass.

## Run

`!bash scripts/kaggle/run_imagenet100_hex_direction_pyramid_2xt4_e20.sh`

20epochs with20-epoch LR schedule,2warmup epochs, seed0, batch256/GPU,2GPUs.
No full training launched locally. This model is still smaller than the5.509M
GE baseline and differs in tokenizer, angle domain, normalization and readout;
the experiment does not isolate Hex geometry alone. The earlier smaller-budget
statement applies to the historical256-width version; the current288/FFN8.25
version approximately matches GE's parameter count.
