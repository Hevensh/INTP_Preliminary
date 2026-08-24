# Mixed-geometry component boundary

The tokenizer is now split into three responsibilities:

1. `geometries.py` owns support extraction and prototype rendering. It does not
   know how poses are selected or how values are aggregated.
2. `cover.py` owns fixed-compatible learnable spatial weighting. It remains a
   `ParameterDict`, so existing checkpoint keys do not change.
3. `state.py` defines the neutral `PrototypeState` and future
   `ComponentAssignment` records used to differentiate trained bases.
4. `rgb_patch_detrend.py` optionally canonicalizes every circular support with
   cosine-weighted RGB moments. It removes one shared RGB mean and the six
   signed RGB-by-xy linear trends, divides the residual by
   `sqrt(weighted_variance + 1e-4)`, and adds only six learned trend V vectors.
   The shared mean (but not the safe sigma) multiplies the geometry-aggregated
   V. This path is opt-in so historical checkpoints remain unchanged.
5. `rgb_patch_standardize.py` is the lighter alternative: it removes only one
   cosine-weighted shared RGB mean, divides by the shared safe standard
   deviation, and leaves all spatial trends intact. In the matching value mode
   each prototype owns separate mean and standard-deviation V vectors in
   addition to its two scale vectors and cosine/sine pair. The six-vector bank
   is initialized as a per-base orthogonal frame; no mean is multiplied back
   into the geometry output.

`MixedGeometryDistanceProjection` remains the orchestration and routing layer.
Its `prototype_states()` method exposes every base through the same schema:
family, scale freedom, direction freedom, active pose count, and null score.

## RoPE-tied value bank

`rope_shared_scale_affine_harmonic` stores only two high-dimensional vectors
for each directional prototype: one scale vector and one direction vector. A
fixed quarter-turn operator `J([a, b]) = [-b, a]` derives each orthogonal
partner. Direction values are `cos(theta) V + sin(theta) J(V)`. Each scale
learns only two scalar mixing coefficients over `Vscale` and `J(Vscale)`.
Color and other non-directional families store only the scale vector. The
conversion from dense pose values uses a constrained complex least-squares
projection, and the progressive split transports shape-compatible AdamW state.

## Next differentiation step

A later planner can consume validation statistics plus `prototype_states()` and
emit assignments such as:

- `full -> radial` when direction responses are consistently tied;
- `full -> angular` when radial behavior is scale-invariant;
- `stripe -> sparse_stripe` when only a stable direction subset is active;
- `any -> color` when spatial variation is negligible;
- retain the source component when state evidence is ambiguous.

Assignments should be applied by copying or projecting parameters into new
components only after the decision is made. This preserves a common trained
starting point and makes every split reversible.
