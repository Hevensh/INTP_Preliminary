# Full-circle first/second moments, four direct-sum groups

PE-only DeiT-Tiny backbone,192 output channels. Keep K24/K12 and polar r5,
but use192 prototypes and14 full-circle directions (2pi*d/14).
Two scale scores are summed BEFORE pose/null softmax, preserving the established
tokenizer order. Null starts at0; real pose probabilities are not renormalized
after dropping null. No Look or differentiation.

For each prototype compute [C1,S1,C2,S2]. Flatten each contiguous group of48
prototypes into192 values, then sum the four groups elementwise. Do NOT average,
concatenate into the backbone, or add a learned projection. Existing output bias
is retained. No zeroth/fourth moment. Opposite peaks survive in second order;
fourfold symmetry and cross-group cancellation are still possible.

Tests compare grouped versus ungrouped outputs and prototype/null gradients,
including prototype chunks crossing group boundaries. CUDA224px AMP full-model
forward/backward passes. No full training performed.

Approximate matching work is4x the96-prototype half7 setup; downstream ViT shape
is unchanged. T4 full-batch memory/throughput is not validated by small local tests.

Run20epochs,20-epoch LR horizon, LR5e-4,2warmup, batch256/GPU,2GPUs, seed0:

`!bash scripts/kaggle/run_imagenet100_full14r5_moments12_b192_sum4_pe_2xt4_e20.sh`
