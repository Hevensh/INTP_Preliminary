# Full6d3r: two independent20-epoch experiments

UPDATE: PE-only has96 prototypes and uses ONLY
first-order cos/sin, no grouping: `rot_hex_full6r3_moment1_b96_pe_ddp_e20.json`
(5,463,556 parameters). Pyramid uses144 prototypes and retains raw six-direction
responses, no moment or input projection, directly feeding144/288/336 stages
(5,510,244 parameters). The abandoned b96 projection variant is not used. Both AMP backward tests
pass. Old192-prototype dual-moment config below is historical, not recommended.

Current commands:

`!bash scripts/kaggle/run_imagenet100_full6r3_moment1_b96_pe_2xt4_e20.sh`

`!bash scripts/kaggle/run_imagenet100_hex_direction_pyramid_full6r3_2xt4_e20.sh`

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
