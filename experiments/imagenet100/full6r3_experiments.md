# Full6d3r: two independent20-epoch experiments

Full6 means0/60/120/180/240/300 degrees, NOT the previous half6 angles.
Polar storage r3 and K24/K12 remain unchanged. Both use the same20-epoch LR
horizon, LR5e-4,2warmup epochs, seed0, batch256/GPU,2GPUs. Planned host account:
XV (xiongwutao); no version number or new accuracy has been verified yet.

1. `rot_hex_full6r3_moments12_b192_sum4_pe_ddp_e20.json`:192 prototypes,
   null-softmax, first/second moments, four groups of48 directly summed to192.
   Ordinary DeiT-Tiny PE-only backbone.5,531,044 parameters. Only angular sample
   count changes relative to full12r3; matching work approximately halves.
2. `hex_direction_pyramid_full6r3_ddp_e20.json`:144 polar prototypes output six
   raw direction slots. Pyramid144/288/336, heads3/3/3, FFN4, two blocks/stage,
   token counts195/52/14, two-ring attention. Relative coordinates and direction
   encoding use the SAME full6 grid.5,510,244 parameters. No tokenizer softmax,
   circular projection, absolute PE or Look. Direction count and tensor shapes
   are unchanged relative to half6 pyramid, so do not expect the same halving.

Full6 is closed under60-degree rotations, unlike half6. This alone does not
prove end-to-end equivariance with pixel sampling, finite boundaries and
sublattice selection. Existing variants keep their old meanings/checkpoint shapes.

Both224px AMP forward/backward tests pass. Training has not been launched locally;
the latest reported disappointing results have not been downloaded in this change.
