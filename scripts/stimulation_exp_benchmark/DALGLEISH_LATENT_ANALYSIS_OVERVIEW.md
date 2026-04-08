# Dalgleish Latent Analysis Overview

## 1. What Is The Dataset?

The Dalgleish dataset is a two-photon calcium-imaging photostimulation dataset.
Each session contains repeated photostimulation trials delivered at multiple stimulation doses, together with catch/control trials.

In plain language:

- a small set of neurons is directly photostimulated on each trial
- many other neurons are recorded simultaneously with two-photon imaging
- the dataset therefore contains both:
  - directly targeted neurons
  - non-targeted neurons that reflect broader recruited network activity

Across the validated analysis path used here, the data are organized into session-level trial tables with trial timing, stimulation dose, and neuron-level response summaries derived from the imaging signals.

## 2. What Scientific Question Are We Asking?

The central question is whether population activity after photostimulation has a dependence structure that is richer than a simple Gaussian model.

More specifically, we ask whether low-dimensional latent population states show:

- non-Gaussian dependence
- dependence beyond pairwise-only structure
- and stronger effects in recruited/non-targeted population activity than in targeted-only activity

This is a methods-paper question.
The dataset is used as a real-data stress test for Dynamic Vine Copulas (DVC), not as a standalone neuroscience paper about photostimulation responses.

## 3. How Is The Data Preprocessed?

The validated latent-state analysis uses the following preprocessing choices.

### Trial extraction

- trials are aligned to photostimulation onset using the author-aligned repository path
- the same trial extraction logic is used across the benchmark and follow-up analyses

### Signal

- the main signal is Suite2p `spks`
- this was the strongest and most stable simple signal choice in the focused representation audits

### Response windows

The latent-state backbone uses two response bins per source neuron:

- delayed stim bin: `0.2` to `0.7` seconds
- early post-stim bin: `0.7` to `1.4` seconds

This avoids the weakest immediate-artifact window and keeps a short response trajectory instead of collapsing the trial into one scalar.

### Train/test splitting

- train/test splits are performed within `session x dose`
- all model comparisons are made on the same held-out trials

### Train-only preprocessing

To avoid leakage:

- PCA is fit on training trials only
- winsorization is fit on training trials only and then applied to test trials
- the empirical CDF transform is fit on training trials only and then applied to test trials

## 4. Why Do We Use PCA?

Direct copula modeling on many raw neurons is too sample-limited in this dataset at the per-slice trial counts available here.

Earlier audits showed:

- raw-neuron formulations become unstable quickly as neuron count grows
- scientifically better windows and temporal bins help
- but the cleanest viable formulation is low-dimensional latent population-state modeling

PCA is therefore used to compress activity from many neurons into a small number of latent coordinates.

Operationally, one latent coordinate means:

- a trial’s score on a train-fit population activity mode

So this analysis is:

- DVC on latent population states derived from many neurons
- not DVC on the original neurons directly

## 5. What Analyses Are Performed?

### Full vine versus baselines

The main static analysis compares full vine against the usable baselines in the latent-state setup:

- Graphical Lasso
- Gaussian SSM
- Gaussian copula
- `1`-truncated vine

TVGL was attempted in the latent-static adaptation, but it did not yield usable session-level outputs and is documented as a feasibility limit rather than silently ignored.

### Pairwise versus higher-order decomposition

The gain is decomposed into:

- pairwise non-Gaussian gain:
  - `NLL(Gaussian) - NLL(1-trunc)`
- higher-order gain:
  - `NLL(1-trunc) - NLL(full)`
  - also referred to as `TC_higher`

### Source-space comparison

The latent source space is compared across:

- targeted-only neurons
- mixed targeted + non-targeted neurons
- non-targeted neurons

This asks where the latent DVC signal lives biologically.

### Family / dependence-type analysis

The selected pair-copula families are summarized both as raw stable families and as grouped publication-facing classes:

- independence
- Gaussian-like elliptical
- heavy-tailed elliptical
- lower-tail asymmetric

### Secondary analyses

Dose, control/catch, and dynamic/history analyses exist in the project outputs, but they are currently secondary to the main static latent claim.

## 6. What Hypotheses Were Tested?

The main latent-state follow-up tested the following hypotheses:

1. Full vine beats Gaussian baselines in latent population space.
2. The gain is not only pairwise-flexible and non-Gaussian, but also includes a higher-order / conditional component.
3. The signal is stronger in recruited/non-targeted or mixed latent space than in targeted-only space.
4. The non-Gaussian structure is not mostly Gaussian-like; it should instead look heavy-tailed and/or asymmetric if DVC is capturing something meaningful.

## 7. What Did We Find?

The strongest current result is the static latent-state result.

In the validated latent-state setup:

- full vine beats Gaussian copula
- full vine beats `1`-truncated vine
- full vine beats Graphical Lasso
- full vine beats Gaussian SSM

The gain decomposes into two positive parts:

- a positive pairwise-flexible / non-Gaussian component
- and a positive higher-order / conditional component

Biologically, the strongest signal is in:

- non-targeted latent space
- or mixed targeted + non-targeted latent space

Targeted-only latent space is clearly weaker.

The family analysis suggests that the latent dependence is driven mainly by:

- heavy-tailed elliptical structure
- plus lower-tail-asymmetric structure

with very little Gaussian-like family usage.

## 8. What Can We Claim, And What Can We Not Claim?

### What we can claim

We can support the following claim:

- in this dataset, low-dimensional latent population states derived from many neurons show non-Gaussian dependence beyond Gaussian baselines
- and they also retain a higher-order / conditional component beyond a pairwise-only truncated vine

We can also say that:

- the strongest latent DVC signal is in recruited/non-targeted or mixed population-state space, not targeted-only space

### What we cannot claim

This is **not** a neuron-level circuit-motif result.

So this analysis does **not** identify:

- specific neuronal triplets or motifs
- direct synaptic mechanisms
- cell-type-specific circuit interactions

Also:

- behavior is not included in the current validated path
- dynamic/time-history analyses exist but are not part of the main claim

## 9. What Figures And Outputs Are Produced?

Main publication-facing figure:

- `results/stimulation_exp_benchmark/plots/fig_latent_publication_final.png`

Supplement-style figures:

- `results/stimulation_exp_benchmark/plots/fig_latent_publication_dose_supplement.png`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_family_supplement.png`

Main summary tables:

- `results/stimulation_exp_benchmark/data/latent_publication_static_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_stats_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_family_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_baseline_feasibility.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_pc_summary.csv`

Figure-to-table mapping:

- `results/stimulation_exp_benchmark/data/latent_publication_figure_panel_map.json`

## 10. How To Rerun The Analysis

The full workflow has three parts:

1. generate the dataset tables and trial-aligned outputs
2. run the latent analysis stack
3. refresh the publication-facing figures

### Step 1. Generate The Dataset Tables

If `dvc_ready/` or the intermediate trial tables are missing, generate them first from the raw Dalgleish session folders:

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp \
python scripts/stimulation_exp_benchmark/build_dalgleish_dvc_dataset.py \
  --data_root dataset_stimulation \
  --out_root dvc_ready \
  --d 10 \
  --seed 0 \
  --selection_mode topk_responsive
```

This writes the base extracted tables and metadata, including:

- `dvc_ready/trial_table.csv`
- `dvc_ready/benchmark_table.parquet` when later benchmark steps are run
- neuron lookup tables and metadata JSON files

What this step assumes:

- the raw Dalgleish dataset is available under `dataset_stimulation/`
- session directories and the author-aligned sidecar files are present locally

If those raw files are missing, the latent analysis scripts cannot regenerate the benchmark by themselves.

### Step 2. Run The Maintained Analysis Script

The maintained end-to-end latent analysis is:

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp \
python scripts/stimulation_exp_benchmark/run_dalgleish_latent_publication_analysis.py \
  --data_root dataset_stimulation \
  --out_root dvc_ready \
  --results_root results/stimulation_exp_benchmark \
  --family_variant stable \
  --seed 0 \
  --n_repeats 2
```

This is now the single supported analysis entrypoint for the Dalgleish latent-state workflow.

Older intermediate scripts used during model selection, follow-up, and debugging have been archived under:

- `scripts/debug_stimulation_exp/`
  This directory is only for archived exploratory/debug scripts. The maintained Dalgleish pipeline is self-contained in `scripts/stimulation_exp_benchmark/`.

### Step 3. Refresh The Publication Figures

The figure-refresh pass that redraws the publication-facing figures from the validated output tables is:

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp \
python scripts/stimulation_exp_benchmark/refresh_dalgleish_latent_publication_figures.py \
  --results_root results/stimulation_exp_benchmark \
  --out_root dvc_ready
```

### One-Script Run Book

For a compact run book that lists the commands by stage and can also execute them, see:

- `scripts/stimulation_exp_benchmark/run_stimulation_exp_benchmark.sh`

For a shorter reproduction summary, see the corresponding section in `README.md`.
