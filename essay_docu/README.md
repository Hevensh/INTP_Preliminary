# SHARE-ViT paper workspace

This directory contains the evidence-bounded manuscript for **SHARE-ViT:
Shared Hexagonal Angle-Scale Routing Enhancement for Vision Transformers**.
Repository code and stored run artifacts remain the source of truth.

## Manuscript files

- `main.tex` / `main.pdf` — single-column LaTeX manuscript and its saved rendering.
- `paper_draft.md` — complete English first draft.
- `claim_ledger.md` — which claims are code-validated, artifact-validated,
  provisional, or prohibited.
- `experiment_map.md` — exact mapping from paper terminology to code, runs, and
  figures.
- `references.bib` — primary literature records used by the draft.
- `imagenet100_core_results_table.tex` / `.pdf` — publication-style main result
  table.
- `figs/` — only the final vector PDF figures used by the manuscript.
- `figure_sources/architecture/` — editable source for the assembled architecture.
- `figure_sources/architecture_components/` — editable component sources and
  intermediate SVG/PNG/PDF renders used to assemble the architecture.
- `figure_sources/legacy_and_supplementary_figures/` — preserved unused,
  comparison, and raster figure variants excluded from the current manuscript.

The LaTeX draft imports three approved vector figures directly from local
`figs/`: the full SHARE-ViT architecture, shared Hex angle-scale geometry,
and half6d3r Look fields. Editable sources, raster previews, comparison panels,
and assembly intermediates stay under `figure_sources/`.

## Current evidence boundary

The main tokenizer/position result is an eight-row, schedule-aligned
ImageNet-100 ablation. All models were trained from scratch for 20 epochs with
the same data and optimizer recipe. The best half6d3r PE+Image-Look model
reaches 55.04% top-1 and 81.80% top-5 with 5.491M parameters; the standard
DeiT-Tiny-sized baseline reaches 51.52% and 79.12% with 5.544M parameters.

The PE + Center Look sharing ablation contains completed G=1/2/3/4/6/12 runs;
G=4 reaches 55.18% top-1 with 5.469M parameters. A separate dual-Look sweep
shows a different optimum: PE + Image Look + G3 Center Look reaches 55.54%
top-1 and 81.82% top-5 with 5.497M parameters. This placement-by-span
interaction is based on one seed and does not establish statistical
significance; the optimized G4 dual-Look point remains to be rerun.

This supports a controlled architectural result, not yet a universal or
statistically conclusive benchmark claim. Multi-seed, longer-schedule,
angle/scale generalization, and downstream-task experiments remain future work.

## Git synchronization

The manuscript, saved PDF, bibliography, figures, editable figure sources and
analysis notes are versioned. Downloaded third-party papers, temporary render
files, LaTeX caches, and legacy/candidate figure archives remain local.
This is a snapshot of the existing draft, not a claim that its results have
been updated to include every newer experiment. `main.tex` is the current
manuscript; older Markdown drafts and evidence notes may lag behind it.

## Evidence updates

Before promoting any new number or conclusion into `paper_draft.md`, add its
artifact path and boundary to `claim_ledger.md`. Incomplete runs and visually
interesting diagnostics must remain clearly labeled.
