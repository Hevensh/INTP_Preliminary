# WV21 Hex pyramid versus historical local GE: code audit

WV21 completed20epochs: best Top1=53.34%, Top5=80.58% at epoch20.
Historical local GE recorded in runs/README:56.06%/82.06%. Gap2.72/1.48pp.
This is NOT a square-versus-Hex-only experiment. No new causal ablation was run.

Matched:144/288/336 widths,2/2/2 depths,3heads,FFN4,~5.51M parameters.

Important differences, verified in model/gevit_tiny.py and hex_direction_vit.py:

1. Readout: GE classifies each position/orientation then spatial SUM and direction
   MAX; Hex uses spatial/direction MEAN then linear classifier. Averaging weak
   directions can dilute isolated strong responses. Spatial sum also changes logit
   scale, so gains cannot be attributed to max alone without an ablation.
2. Norm: GE GroupNorm(1,C) on B,C,G,H,W normalizes C/G/H/W per image. Hex LayerNorm(C)
   normalizes each token/direction separately. This removes local response mean/
   scale, which may matter for raw matching; that is a hypothesis, not proof.
3. Downsampling: GE2x2 max-pool then1x1 channel projection. Hex selects even/even
   axial sites after attention, then projects. Learned attention is not guaranteed
   to preserve peaks before decimation. No dedicated pooling/anti-alias stage.
4. Initialization: GE Conv/Linear truncated normal std.02, bias0; Hex defaults.
5. Attention temperature: GE divides by sqrt(total C), Hex SDPA by sqrt(C/3).
   For identical unscaled scores Hex logits would be sqrt(3) larger; actual
   distributions also depend on norm/initialization/PE and cannot be inferred.
6. PE: GE concatenates row/column/group features; Hex adds two learned2D MLP
   embeddings over query-frame position and direction cos/sin. Not identical PE.
7. First layer: GE Cartesian16x16 rotated90 degrees; Hex r3 polar K24/K12
   compensated scale-sum,60-degree rendered poses and different spatial sampling.
8. Space:196/49/9 square versus195/52/14 Hex; two rings25 versus19neighbors.

Prioritize isolated readout and norm/initialization/temperature alignment before
concluding Hex is inferior. A fully aligned engineering control may combine these,
but cannot identify the contribution of each change. Tokenizer shape and60-degree
pixel interpolation should be evaluated only after controlling backbone choices.

The separate full12r3 first-moment96-base PE experiment changes ONLY the angular
sampling count relative to full6r3, keeping the null candidate and all other
training settings. It does not eliminate opposite-peak cancellation: twelve
directions also include opposing pairs. It tests sampling density, not a remedy
for first-moment information loss.
