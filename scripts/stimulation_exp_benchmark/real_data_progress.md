# Real-Data Progress

## Objective Of This Rerun

Run the full stable 11-session Dalgleish rerun on the three prioritized static variants, choose the single best construction for the real-data section, and then run one compact within-session temporal check on that best variant only.

## What Was Held Fixed

These points were treated as established from the previous debug pass and were not reopened:

- no dataset rediscovery,
- stable family restriction only,
- same train-only neuron selection, demeaning, winsorization, and empirical-CDF mapping policy,
- same repository Gaussian, 1-truncated vine, and full-vine fit path,
- no return to the unstable default family set for the main rerun,
- dynamic analysis deferred until after choosing the best static construction.

## Benchmark Variants Run

The full 11-session static rerun used exactly these three variants:

| variant | feature | d | targeted policy | selection | residualization |
|---|---|---:|---|---|---|
| `variant_A_stim_mean_d4` | `stim_mean` | 4 | exclude targeted | top-k responsive | yes |
| `variant_B_stim_mean_d6` | `stim_mean` | 6 | exclude targeted | top-k responsive | yes |
| `variant_C_diff_d4` | `stim_mean - baseline_mean` | 4 | exclude targeted | top-k responsive | yes |

All three variants used repeated train/test splits within `session x dose` with the stable family restriction.

## Code Changes Made

- Added [scripts/run_dalgleish_real_data_decision_rerun.py](/Users/alessandro/Documents/github/DVC/scripts/run_dalgleish_real_data_decision_rerun.py) to run the full stable decision rerun, write the updated benchmark artifacts, choose the best variant, and perform the compact dynamic follow-up.
- Kept [scripts/run_dalgleish_real_data_benchmark.py](/Users/alessandro/Documents/github/DVC/scripts/run_dalgleish_real_data_benchmark.py) as the underlying validated helper path.
- Preserved the stable family default introduced in the previous pass.

## Full-Session Results

Each variant produced `231` usable static `session x dose x repeat` slices across all `11` sessions.

Variant summary:

| variant | Gaussian mean NLL | 1-trunc mean NLL | full mean NLL | mean full delta vs Gaussian | prop full < Gaussian | prop full < 1-trunc | mean TC_higher | median TC_higher | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `variant_A_stim_mean_d4` | -0.054 | -0.049 | -0.030 | -0.024 | 0.506 | 0.532 | -0.019 | 0.012 | `usable_but_heterogeneous` |
| `variant_B_stim_mean_d6` | 0.051 | -0.073 | 0.086 | -0.034 | 0.433 | 0.377 | -0.158 | -0.091 | `usable_but_heterogeneous` |
| `variant_C_diff_d4` | 0.042 | 0.018 | 0.121 | -0.079 | 0.368 | 0.398 | -0.102 | -0.024 | `not_useful` |

Interpretation:

- `variant_A_stim_mean_d4` is the best stable construction on the full 11-session rerun.
- `stim_mean` is clearly better than `stim_mean - baseline_mean` for this benchmark.
- Lower dimension helps materially: `d=4` is much more competitive than `d=6`.
- Even in the best variant, the average full-vine gain is slightly negative, but it is very close to neutral.
- The best variant is heterogeneous rather than decisively favorable or decisively unfavorable.

## Best Variant Detail

The selected best variant was:

- `variant_A_stim_mean_d4`

Key properties of the best variant:

- full vine beats Gaussian in `50.6%` of static slices,
- full vine beats 1-trunc in `53.2%` of static slices,
- mean `TC_higher = -0.019`,
- median `TC_higher = +0.012`.

Dose-level behavior for the best variant:

| dose | full delta vs Gaussian | 1-trunc delta vs Gaussian | mean TC_higher |
|---:|---:|---:|---:|
| 5 | 0.035 | -0.031 | 0.066 |
| 10 | 0.036 | 0.030 | 0.006 |
| 25 | -0.001 | 0.070 | -0.071 |
| 50 | -0.034 | -0.038 | 0.004 |
| 75 | -0.052 | -0.010 | -0.042 |
| 100 | -0.042 | 0.037 | -0.079 |
| 200 | -0.110 | -0.092 | -0.017 |

Interpretation:

- The best variant is **not** a convincing dose-based higher-order win.
- It is closest to useful at low doses and near-neutral overall.
- At stronger doses, neither full vine nor 1-trunc improves systematically over Gaussian in the way we would want for a clean main methods figure.
- The result is best described as weak, heterogeneous, and only modestly suggestive.

## Session Robustness

The best variant is broad enough to be nontrivial, but not clean:

- some sessions are clearly unfavorable for the full vine, such as `20191010_L542` and `20191129_OG423`,
- several sessions are mildly favorable or near-neutral,
- the favorable sessions are not strong enough to drive a convincing aggregate win.

Examples of per-session mean full-vine delta vs Gaussian for the best variant:

- `20191010_L542`: `-0.231`
- `20191129_OG423`: `-0.153`
- `20191203_L541`: `+0.077`
- `20191127_L543`: `+0.044`
- `20191202_L543`: `+0.042`

Interpretation:

- The result is not dominated by a single session.
- It is also not consistent enough across sessions for a strong “DVC wins on real data” claim.

## Dynamic Follow-Up

The dynamic check was run **only** for the best variant:

- `variant_A_stim_mean_d4`

Dynamic setup:

- strongest eligible dose per session,
- early / middle / late trial-order blocks,
- train/test split within each block,
- same train-only preprocessing and stable family restriction.

Dynamic coverage:

- `10` sessions contributed usable dynamic blocks,
- all dynamic slices were at dose `200`,
- `dynamic_summary.csv` contains `90` rows (`10` or `9` session-block combinations x `3` models).

Mean dynamic results by block:

| block | full mean NLL | full delta vs Gaussian | truncated delta vs Gaussian | mean TC_higher |
|---|---:|---:|---:|---:|
| early | 0.396 | 0.200 | 0.407 | -0.207 |
| middle | 1.214 | -0.190 | 0.513 | -0.702 |
| late | 0.690 | 0.126 | 0.257 | -0.183 |

Interpretation:

- There **is** meaningful variation over trial order under the best stable construction.
- The temporal axis is therefore best classified as:
  - `credible_temporal_axis`
- However, it is **not** a clean dynamic full-vine win.
- The strongest systematic signal is that model behavior changes over trial order, especially a poor middle block for full vine, not that higher-order gains steadily emerge over time.

## Outputs Generated

Updated artifacts written in this rerun:

- [benchmark_table.parquet](/Users/alessandro/Documents/github/DVC/dvc_ready/benchmark_table.parquet)
- [metrics_table.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/metrics_table.csv)
- [variant_summary.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/variant_summary.csv)
- [dynamic_summary.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/dynamic_summary.csv)
- [neuron_lookup.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/neuron_lookup.csv)
- [fig_realdata_fullrerun_variant_comparison.png](/Users/alessandro/Documents/github/DVC/fig_realdata_fullrerun_variant_comparison.png)
- [fig_realdata_best_variant_dose_summary.png](/Users/alessandro/Documents/github/DVC/fig_realdata_best_variant_dose_summary.png)
- [fig_realdata_best_variant_session_robustness.png](/Users/alessandro/Documents/github/DVC/fig_realdata_best_variant_session_robustness.png)
- [fig_realdata_best_variant_dynamic_check.png](/Users/alessandro/Documents/github/DVC/fig_realdata_best_variant_dynamic_check.png)

## Paper Recommendation

### Main Figure / Supplement / Drop

Recommended outcome for the Dalgleish dataset:

- `Outcome B: supplement / control example`

Reason:

- After the corrected stable full-session rerun, the dataset does **not** give a convincing real-data full-vine win.
- The best variant is close to neutral and heterogeneous rather than clearly positive.
- This makes the dataset hard to use as the paper’s main real-data figure if the paper needs a clear “full vine explains more” message.
- It is still useful as a real-data control example showing that:
  - the corrected full-vine benchmark is numerically sane,
  - the best construction is only weakly favorable,
  - and strong intervention data do not automatically produce a full-vine win.

### Methods-Paper Usefulness

The most defensible use is:

- as a supplement or control example,
- emphasizing that the real-data result is mixed / heterogeneous,
- and that the dataset is informative precisely because it does **not** hand the method an easy win.

### Temporal Axis Recommendation

Recommended temporal classification:

- `credible_temporal_axis`

with the important caveat:

- the temporal signal is useful for showing that dependence behavior changes over trial order,
- but it does **not** currently strengthen the case for full-vine superiority.

## Final Plain-Language Conclusions

1. **Which variant is best on the full 11-session rerun?**
   `variant_A_stim_mean_d4`.

2. **Is Dalgleish good enough for the main real-data figure, supplement, or neither?**
   Best current recommendation is **supplement / control example**, not the main real-data figure.

3. **Does the dataset show a meaningful within-session temporal axis under the best variant?**
   Yes. There is a **credible temporal axis**, but it is not a clean dynamic win for the full vine.

## Representation / Artifact Audit

### Objective Of This Pass

Give the Dalgleish dataset one final focused chance by testing whether the remaining weakness comes from response representation, photostimulation-artifact contamination, signal-type choice, targeted-neuron handling, or a few poor sessions washing out a stronger subset.

### Hypotheses Tested

- The original `stim_mean` window may be too close to the laser period.
- A later or post-stim response window may be cleaner.
- Scalar averaging may wash out higher-order dependence that a tiny time-binned representation can preserve.
- Raw fluorescence or neuropil-corrected fluorescence might outperform the current deconvolved-like signal.
- Mixed targeted/non-targeted subsets might reveal more relevant interactions than pure exclusion.
- A few poor sessions might be suppressing a stronger subset.

### Representation Variants Run

All variants kept the stable family restriction, train-only preprocessing, and the same repository fit/score path.

Window / artifact audit on the validated base setup:

- `window_current_stim_mean`
- `window_delayed_stim_mean`
- `window_post_stim_mean`
- `window_stim_skip_early`

Signal-type audit using the best window from the previous step:

- `signal_spks_window_post_stim_mean`
- `signal_F_window_post_stim_mean`
- `signal_Fcorr_window_post_stim_mean`

Neuron-policy audit using the best scalar signal/window:

- `policy_exclude_spks_post_stim_mean`
- `policy_include_spks_post_stim_mean`
- `policy_mixed_spks_post_stim_mean`

Tiny time-binned representation:

- `binned_spks_post_stim_mean`

### Artifact / Window Checks

Window audit summary:

| variant | mean full delta vs Gaussian | prop full < Gaussian | mean TC_higher |
|---|---:|---:|---:|
| `window_current_stim_mean` | -0.038 | 0.506 | -0.028 |
| `window_delayed_stim_mean` | -0.006 | 0.515 | 0.036 |
| `window_stim_skip_early` | -0.003 | 0.498 | 0.010 |
| `window_post_stim_mean` | 0.001 | 0.498 | 0.050 |

Interpretation:

- Yes, the response window matters.
- The current `stim_mean` window was the weakest of the four tested windows.
- Moving later in time helped.
- The best scalar window was the early post-stim window, not the original stim window.

### Signal-Type Checks

Signal audit at the best scalar window:

| signal | mean full delta vs Gaussian | prop full < Gaussian | mean TC_higher |
|---|---:|---:|---:|
| `spks` | 0.006 | 0.519 | 0.051 |
| `F` | -0.096 | 0.403 | -0.019 |
| `Fcorr = F - 0.7 * Fneu` | -0.125 | 0.351 | -0.052 |

Interpretation:

- The best signal remained `spks`.
- Raw fluorescence and neuropil-corrected fluorescence were both materially worse.
- So the main problem was **not** that the benchmark was using the wrong signal family. The current deconvolved-like signal was already the strongest of the available simple choices.

### Neuron-Policy Checks

Targeted-policy audit at the best scalar signal/window:

| policy | mean full delta vs Gaussian | prop full < Gaussian | prop full < 1-trunc | mean TC_higher |
|---|---:|---:|---:|---:|
| exclude targeted | -0.001 | 0.489 | 0.489 | 0.046 |
| include targeted | 0.005 | 0.489 | 0.506 | 0.023 |
| mixed targeted/non-targeted | 0.001 | 0.506 | 0.519 | 0.030 |

Interpretation:

- Targeted policy changed the result only modestly.
- Including or mixing targeted neurons did **not** materially rescue the story.
- The dependence signal is not obviously hidden in a simple targeted/non-targeted mixture effect.

### Tiny Time-Binned Representation

This was the strongest last-chance representation:

- `binned_spks_post_stim_mean`

It used:

- `spks`
- two bins across the late stim / early post period
- total dimension `4` via `2` selected neurons x `2` bins
- targeted exclusion

Static results for this best last-chance variant:

| metric | value |
|---|---:|
| Gaussian mean NLL | -0.130 |
| 1-trunc mean NLL | -0.067 |
| full mean NLL | -0.169 |
| mean full delta vs Gaussian | 0.039 |
| prop full < Gaussian | 0.528 |
| prop full < 1-trunc | 0.597 |
| mean TC_higher | 0.102 |
| median TC_higher | 0.037 |

Interpretation:

- This is a real improvement over the previous best full-session result.
- The best full-session result before this audit was near-neutral and slightly negative on average.
- The binned post-stim `spks` representation makes the full vine modestly favorable on average and more favorable relative to 1-trunc.
- So the main weakness was **partly** the representation/window choice.

### Session-Quality Checks

Session-quality table:

- [session_quality_audit.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/session_quality_audit.csv)

Key observations:

- `7` sessions were flagged `possible_low_quality`
- `3` were flagged `higher_quality`
- `1` was `typical`

Exploratory subset check using the non-`possible_low_quality` sessions only:

- subset sessions: `4`
- mean full delta vs Gaussian: `-0.029`
- prop full < Gaussian: `0.429`
- prop full < 1-trunc: `0.619`
- mean `TC_higher`: `0.051`

Interpretation:

- A simple quality screen does **not** reveal a cleaner subset that strengthens the full-vine case.
- In fact, the modest positive full-vine gain in the best last-chance variant does not improve under this coarse quality filter.
- So the dataset is not being held back mainly by a few obviously poor sessions.

### Dynamic Follow-Up Under The Best Last-Chance Variant

The best audited variant still showed a credible temporal axis:

- `credible_temporal_axis`

But the dynamic pattern did **not** become more interpretable in a DVC-favorable direction.

Mean dynamic results for `binned_spks_post_stim_mean`:

| block | full delta vs Gaussian | truncated delta vs Gaussian | mean TC_higher |
|---|---:|---:|---:|
| early | -0.326 | -0.070 | -0.256 |
| middle | 0.062 | 0.158 | -0.096 |
| late | -0.108 | 0.190 | -0.270 |

Interpretation:

- Trial-order dependence still changes over time.
- But the clearer pattern is that model behavior varies over blocks, not that full vine becomes consistently stronger.
- So the temporal axis remains real but not more supportive of the main DVC narrative.

### Results

Main conclusions from the last-chance audit:

- The response representation/window **did** matter.
- Moving from the original stim window to the early post-stim window improved the scalar result.
- A tiny binned post-stim `spks` representation improved the full-vine case further and gave the best audited result.
- Signal choice did **not** rescue the result: `spks` remained best.
- Targeted-neuron policy only changed the result modestly.
- A coarse session-quality screen did **not** reveal an obviously stronger subset.

### Final Recommendation

Representation-audit outcome:

- `Outcome B: improved but still heterogeneous`

This is better than the pre-audit conclusion, but not enough to change the paper recommendation to a clear main-text real-data win.

The most defensible updated reading is:

- the dataset is **supplement-worthy / control-worthy**
- representation matters,
- a cleaner late-response representation can make the full-vine result modestly favorable,
- temporal variation is still credible,
- but the result remains too heterogeneous and too small to be the paper’s main real-data figure.

### Updated Plain-Language Answers

1. **Was the main problem likely the response representation/window?**
   Partly yes. The original stim-window scalar summary was not the best representation, and moving to post-stim plus a tiny binned representation improved the full-vine result materially.

2. **Did any variant materially improve the full-vine case?**
   Yes. `binned_spks_post_stim_mean` improved the mean full-vine delta vs Gaussian to `+0.039` and mean `TC_higher` to `+0.102`, which is meaningfully better than the previous near-neutral baseline.

3. **Does the credible temporal axis become more interpretable under a better representation?**
   Not really. The temporal axis remains credible, but it still does not produce a clean dynamic full-vine win.

4. **Is Dalgleish now main-text worthy, supplement-worthy, or still just a control example?**
   Best current recommendation remains **supplement-worthy / control example**, now with the added note that representation matters and a modestly more favorable full-vine result can be obtained with a cleaner late-response representation.

## Neuroscientific Validity Audit: Hypotheses And Proposed Tests

### Current Benchmark Representation In Plain Language

Here is what the current benchmark is actually doing, stated in neuroscience terms rather than benchmark terms.

- Trials are defined from photostimulation onset times extracted from the `paqanalysis` files.
- The baseline window is currently `-1.0 s` to `-0.1 s` relative to stimulation onset.
- The standard stim window in the builder is `0.0 s` to `1.0 s`.
- The standard post window in the builder is `1.0 s` to `2.0 s`.
- The current “best” last-chance variant does **not** use one scalar per neuron across the whole stim window.
- Instead, it uses the Suite2p `spks` matrix and represents each selected neuron by **two time bins**:
  - late stim bin: `0.2 s` to `0.7 s`
  - early post bin: `0.7 s` to `1.4 s`
- In that best audited variant, the model sees:
  - `2 neurons x 2 time bins = 4 dimensions total`
- So each dimension is not “a neuron” anymore. It is a specific `neuron x time-bin` feature.

This means the current best result is based on a very compressed representation of the response:

- only `2` neurons are used,
- each neuron is summarized by `2` coarse temporal bins,
- and the benchmark therefore asks a dependence question in a very small subspace.

### What Is Biologically Questionable Right Now

The main neuroscientific concerns are:

- Using only `2` neurons may make the higher-order dependence question nearly impossible by construction.
- A coarse `2-bin` summary may be too compressed to represent the actual photostimulation response dynamics.
- The current windows may still not align with the biologically relevant recruitment latency.
- The current dynamic blocks may reflect adaptation, bleaching, state drift, or selection instability rather than meaningful dependence evolution.
- The current benchmark mixes sessions with very different response magnitudes and responsive fractions.
- We may be asking the wrong dependence question biologically: higher-order structure might live in targeted-to-recruited interactions or short-latency temporal coordination rather than in a tiny low-dimensional summary of recruited-neuron amplitudes.

### Hypotheses

#### H1. Neuron-count restriction is too severe

The current best representation may be too small to fairly test higher-order dependence, because `2 neurons x 2 bins` gives the full vine very little room to express genuinely higher-order structure.

What would support this:

- larger neuron subsets with sensible representations produce more interpretable or more stable full-vine gains.

What would weaken this:

- larger neuron subsets remain no better or become clearly worse even under neuroscientifically sensible representations.

#### H2. The response windows still do not match the relevant neural latency

The recruited network response may occur later than the current summary captures, or may extend over a longer integration window.

What would support this:

- later, longer, or multi-bin windows produce more coherent response structure and better full-vine behavior.

What would weaken this:

- biologically plausible alternative windows yield essentially the same result.

#### H3. Scalar summaries are still too compressed

Even after the last audit, the best summary may still be too coarse. A richer within-trial temporal representation could preserve coordination structure that scalar averaging discards.

What would support this:

- 3-bin or 4-bin representations outperform 1-bin and 2-bin summaries in a stable, interpretable way.

What would weaken this:

- adding temporal detail does not help, or only helps by making the problem numerically easier in a non-biological way.

#### H4. The scientifically relevant signal may depend on neuron class

The most meaningful dependence may live in:

- targeted neurons only,
- recruited non-targeted neurons only,
- or explicitly mixed targeted-to-recruited subsets.

What would support this:

- one of these policies yields much more coherent full-vine or pairwise-vs-higher-order behavior.

What would weaken this:

- all three policies behave similarly after controlling representation.

#### H5. The current dynamic analysis may be conflating trial-order effects with physiology-irrelevant drift

Early/middle/late block differences may reflect bleaching, slow state drift, or unstable neuron selection rather than stimulation-related dependence changes.

What would support this:

- the dynamic signal tracks response amplitude decay, baseline drift, or changing neuron identities more than dependence-specific quantities.

What would weaken this:

- dynamic differences remain after checking baseline level, response magnitude, and neuron-selection stability within session.

#### H6. A few sessions may be biologically unsuitable for this question

Some sessions may have too few strong responses, too few mapped targets, too few responsive neurons, or too little separation between targeted and recruited populations.

What would support this:

- there is a principled subset of sessions with stronger and more coherent response structure.

What would weaken this:

- no reasonable quality screen produces a more interpretable result.

### Proposed Tests

#### Test 1. Neuron-count audit

Goal:

- quantify how many neurons are actually available, how many are targeted, and how many are being used in each representation.

What I would inspect or generate:

- `Fall.mat`-derived ROI counts already exposed through the builder
- target-to-ROI mappings already stored in `targeted_roi_by_program`
- a new table such as `results/stimulation_exp_benchmark/data/neuron_count_audit.csv`

Key outputs:

- total ROIs per session
- usable ROIs after `iscell`
- mapped targeted ROIs
- non-targeted ROIs
- neurons used per variant
- effective “neuron x time-bin” dimensionality

Why this matters:

- it tells us whether the current higher-order question is underspecified by construction.

#### Test 2. Time-window validity audit

Goal:

- determine whether the current summary windows align with plausible calcium / deconvolved response timing.

What I would inspect or generate:

- existing trial timing metadata from `trial_table.csv`
- frame-rate information from `trial_table.csv` and `representation_audit_metadata.json`
- new comparison outputs for biologically plausible windows

Proposed windows:

- immediate stim: around `0.0 s` to `0.5 s`
- delayed stim: around `0.2 s` to `0.7 s`
- early post: around `0.7 s` to `1.4 s`
- longer integration windows of about `500 ms` and `1000 ms`

Why this matters:

- it tells us whether we are averaging over artifact, missing the recruited response, or using an integration timescale that is too crude for calcium dynamics.

#### Test 3. Representation-depth audit

Goal:

- test whether one scalar or two bins is still too compressed.

What I would inspect or generate:

- a comparison of:
  - one scalar per neuron
  - two bins per neuron
  - three or four bins per neuron
- likely a new table such as `results/stimulation_exp_benchmark/data/representation_depth_audit.csv`

Important constraint:

- keep total dimension modest and neuroscience-motivated, not a broad hyperparameter sweep.

Why this matters:

- it directly tests whether temporal compression is washing out coordination structure.

#### Test 4. Neuron-policy audit with more neuroscience meaning

Goal:

- compare targeted-only, non-targeted-only, and mixed targeted-plus-recruited subsets under the **same** temporal representation.

What I would inspect or generate:

- targeted ROI mappings already in the builder outputs
- new policy-specific metrics tables and possibly a targeted-vs-nontargeted neuron audit table

Why this matters:

- the biologically relevant dependence question may be about coupling between perturbed cells and recruited cells, not just the marginal response of a handful of non-targeted neurons.

#### Test 5. Trial-definition / carry-over audit

Goal:

- check whether trial spacing and baseline definitions make sense for calcium responses and whether carry-over is plausible.

What I would inspect or generate:

- `trial_table.csv`
- stimulation-time differences within each session
- baseline/stim/post timing from the builder
- a simple table of inter-trial intervals and possible overlap flags

Why this matters:

- if trials are too close together relative to the calcium decay timescale, then “baseline” and “dynamic” effects may partly reflect carry-over from previous stimulation.

#### Test 6. Session-quality and heterogeneity audit with neuroscience framing

Goal:

- determine whether some sessions are biologically unsuitable rather than just statistically weak.

What I would inspect or generate:

- the existing [session_quality_audit.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/session_quality_audit.csv)
- a refined session-quality table that also includes:
  - targeted-neuron availability
  - inter-trial timing
  - response latency or window-specific response strength
  - neuron-selection stability across variants

Why this matters:

- it tells us whether the dataset should be interpreted as a coherent pooled benchmark or as a heterogeneous collection where only some sessions match the scientific question.

### Exact Files / Scripts I Would Use Or Generate

Existing files to inspect:

- [build_dalgleish_dvc_dataset.py](/Users/alessandro/Documents/github/DVC/build_dalgleish_dvc_dataset.py)
- [scripts/run_dalgleish_real_data_benchmark.py](/Users/alessandro/Documents/github/DVC/scripts/run_dalgleish_real_data_benchmark.py)
- [scripts/run_dalgleish_real_data_decision_rerun.py](/Users/alessandro/Documents/github/DVC/scripts/run_dalgleish_real_data_decision_rerun.py)
- [scripts/run_dalgleish_representation_audit.py](/Users/alessandro/Documents/github/DVC/scripts/run_dalgleish_representation_audit.py)
- [dvc_ready/trial_table.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/trial_table.csv)
- [dvc_ready/neuron_lookup.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/neuron_lookup.csv)
- [dvc_ready/metrics_table.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/metrics_table.csv)
- [dvc_ready/metrics_table_representation_audit.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/metrics_table_representation_audit.csv)
- [dvc_ready/session_quality_audit.csv](/Users/alessandro/Documents/github/DVC/dvc_ready/session_quality_audit.csv)
- [results/stimulation_exp_benchmark/data/representation_audit_metadata.json](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/representation_audit_metadata.json)

Likely new outputs for Stage 2 if approved:

- `results/stimulation_exp_benchmark/data/neuron_count_audit.csv`
- `results/stimulation_exp_benchmark/data/window_validity_audit.csv`
- `results/stimulation_exp_benchmark/data/representation_depth_audit.csv`
- `results/stimulation_exp_benchmark/data/trial_spacing_audit.csv`
- `results/stimulation_exp_benchmark/data/session_quality_audit_refined.csv`
- a small set of corresponding diagnostic plots under `results/stimulation_exp_benchmark/plots/`

### Estimated Scope Of The Run

This would be a medium-sized focused audit, not a broad benchmark sweep.

Expected scope:

- one pass over the existing trial tables and builder metadata
- one compact set of representation comparisons
- one compact neuron-policy comparison
- one compact session-quality / trial-spacing pass

Estimated runtime:

- likely on the order of one longer local run, similar to or somewhat smaller than the last representation audit
- no synthetic suite
- no dataset rediscovery
- no broad family or hyperparameter search

### What Each Test Would Let Us Conclude

- If larger neuron sets and richer temporal representations help: the current analysis is too compressed neuroscientifically.
- If later windows help but richer temporal detail does not: the main issue is timing / artifact contamination rather than representation depth.
- If targeted-only or mixed subsets help: the biologically relevant dependence may be targeted-to-recruited interaction structure.
- If trial-spacing looks too short: current baseline and dynamic interpretations may be contaminated by carry-over.
- If no biologically sensible variant helps: the dataset is probably still best treated as a supplement/control example, and the weakness is not just preprocessing.

## Neuroscientific Validity Audit: Results And Decision

### Objective Of This Pass

This pass was run as a decision-oriented audit, not just a model-comparison exercise.

The question was:

- what is the **smallest neuroscientifically defensible representation** that is still statistically viable for DVC on this dataset?

### What Was Actually Run

The audit preserved the validated benchmark path:

- stable family restriction only,
- same train-only preprocessing and empirical-CDF mapping,
- same trial extraction and held-out scoring policy,
- no dataset rediscovery,
- no broad hyperparameter sweep.

The focused audit then ran:

- neuron-count scaling at `4`, `10`, `25`, and `50` neurons,
- window comparisons with timing reported in both seconds and effective imaging frames,
- representation-depth comparisons across `1`, `2`, and `4` temporal bins,
- signal-type comparisons across `spks`, `F`, and `Fcorr`,
- secondary neuron-policy comparisons across non-targeted, targeted, and mixed subsets,
- and one compact dynamic follow-up under the best audited representation.

### What The Current Sessions Actually Look Like

Across the `11` usable sessions, the builder is not starved for raw ROIs:

- mean usable ROIs per session: about `2809`
- mean mapped targeted ROIs per session: about `130`
- mean non-targeted usable ROIs per session: about `2679`

So the bottleneck is **not** lack of available neurons in the raw data.
The bottleneck is the ratio of:

- trial count per `session x dose` slice
to
- the number of neuron or neuron x time-bin features we are trying to fit.

### Neuron-Count Scaling

The neuron-count scaling study used the same early post-stim `spks` scalar representation for all counts.

Results:

- `4` neurons: fully viable across all `11` sessions and `154` slices
- `10` neurons: still viable across all `11` sessions and `142` slices
- `25` neurons: only `1` session and `2` slices remained usable
- `50` neurons: no usable slices remained

Interpretation:

- Yes, the benchmark becomes **undersampled by construction** once we try to fit direct raw-neuron representations at `25` to `50` neurons.
- The dataset is therefore not suitable for “just use many raw neurons” copula fitting on a per-slice basis.
- The viable neuron-count range for this benchmark is much smaller:
  - about `4` to `10` neurons for scalar summaries
  - or about `3` to `5` neurons when using multiple temporal bins

### Time-Window Validity

The window audit was run at an effective frame rate of about `7 Hz`, so the tested windows correspond to:

- `0.0–0.5 s`: about `4` frames
- `0.2–0.7 s`: about `4` frames
- `0.7–1.4 s`: about `5` frames
- `0.2–1.2 s`: about `7` frames
- `0.0–1.0 s`: about `7` frames

The best scalar window was the **early post-stim window**:

- `0.7–1.4 s`
- about `5` imaging frames after stimulation onset

Interpretation:

- The original broad stim window was the weakest of the tested scalar windows.
- Delaying the response summary improves the benchmark materially.
- This supports the concern that immediate post-onset averaging is not the best biological summary here, either because of stimulation artifact, latency of recruited responses, or both.

### Representation Depth

Three controlled temporal representations were compared:

- scalar summary:
  - `10 neurons x 1 bin = 10 dimensions`
  - each feature means: “one neuron’s mean `spks` from `0.2 s` to `1.4 s`”
- two-bin summary:
  - `5 neurons x 2 bins = 10 dimensions`
  - each feature means: “one neuron’s mean `spks` from `0.2–0.7 s` or `0.7–1.4 s`”
- four-bin summary:
  - `3 neurons x 4 bins = 12 dimensions`
  - each feature means: “one neuron’s mean `spks` in one of four bins: `0.2–0.5`, `0.5–0.8`, `0.8–1.1`, `1.1–1.4 s`”

The human-readable feature mapping is written in:

- [feature_semantics_audit.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/feature_semantics_audit.csv)

Key result:

- the best overall held-out variant was `depth_4bin_3n`
- this means `3 neurons x 4 delayed/post-stim bins = 12 features`

Interpretation:

- Temporal compression is a real bottleneck.
- A richer within-trial response representation helps more than simply increasing raw neuron count.
- The best audited representation is still small, but it is more neuroscientifically interpretable than the previous `2 neurons x 2 bins` construction because it retains a short response trajectory rather than collapsing it immediately.

Important caveat:

- `depth_4bin_3n` improved full vine vs Gaussian,
- but it still had negative mean `TC_higher`,
- so it did **not** become a clean higher-order win over the `1`-truncated vine.

### Signal-Type Audit

At the best scalar window, the signal comparison was clear:

- `spks` was best
- `F` was worse
- `Fcorr = F - 0.7 * Fneu` was also worse

Interpretation:

- The benchmark was not being held back by choosing the wrong simple signal family.
- The current deconvolved-like `spks` representation remains the best of the available straightforward choices.

### Neuron-Class Policy

This was kept secondary, as requested.

The targeted-policy comparison changed the result only modestly:

- targeted-only and mixed subsets slightly improved full-vs-Gaussian relative to non-targeted-only in the tested two-bin setup,
- but none of them produced positive average `TC_higher`.

Interpretation:

- Neuron-class choice matters less than windowing and temporal compression.
- The main bottleneck does **not** appear to be a simple “we excluded the wrong neurons” issue.

### Trial Spacing And Carry-Over

The trial-spacing audit showed:

- session median inter-trial interval around `11.6–18.0 s`
- but minimum inter-trial interval of `0 s` in every session
- only about `0.6%` to `2.2%` of intervals were below `2–3 s`

Interpretation:

- There is a small amount of potential carry-over or duplicate/onset-clustering behavior that deserves a warning flag.
- But the typical spacing is long relative to the short response windows being modeled here.
- So carry-over is a real caveat, but not the dominant explanation for the benchmark behavior.

### Session Heterogeneity

The refined session-quality table is:

- [session_quality_audit_refined.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/session_quality_audit_refined.csv)

What it says in plain language:

- sessions differ a lot in response strength and how favorable they are to the full vine,
- but there is no obvious “drop these few bad sessions and the story becomes clean” subset,
- and the dataset should still be treated as genuinely heterogeneous rather than secretly clean after filtering.

### Dynamic Follow-Up

The best audited static representation was used for one compact dynamic check:

- `depth_4bin_3n`

This confirms that there is still a within-session temporal axis.
But the dynamic result became **less** interpretable for the paper story:

- blockwise fits in this `12`-dimensional representation produced very large Gaussian NLLs in some blocks,
- full vine still did not show a clean higher-order gain over `1`-trunc,
- and the temporal variation is therefore better described as:
  - dependence behavior changes over trial order,
  - but not in a clean “DVC wins dynamically” way.

So the temporal axis remains credible, but not more persuasive for the main claim.

### Direct Answers To The Decision Questions

1. **Is the current analysis too undersampled in neuron count?**
   Yes, if we try to fit direct raw-neuron copulas at `25` to `50` neurons per slice. No, for carefully compressed representations in the `4–12` feature range.

2. **What neuron count range seems scientifically meaningful and still numerically viable?**
   The most defensible range is:
   - about `4–10` neurons for scalar summaries
   - about `3–5` neurons when each neuron contributes multiple temporal bins
   - keeping total feature dimension around `8–12`

3. **Is the main bottleneck neuron count, time window, temporal compression, or neuron-class choice?**
   The main bottlenecks are:
   - first: neuron-count feasibility relative to trial count
   - second: time-window choice and temporal compression
   - third: only distantly, neuron-class choice

4. **What exact next benchmark should we run after this audit?**
   The next benchmark should be:
   - `spks`
   - top-k responsive non-targeted neurons
   - skip the earliest `0.0–0.2 s`
   - use a compact delayed/post-stim temporal representation
   - recommended concrete setup: `4 neurons x 2 bins`
   - bins: `0.2–0.7 s` and `0.7–1.4 s`
   - total dimension: `8`

Why this next benchmark:

- it is still small enough to be statistically viable,
- it is more neuroscientifically defensible than a single scalar per neuron,
- it avoids the weakest immediate-artifact window,
- and it leaves more room for higher-order structure than the previous ultra-compressed `2 neurons x 2 bins` setup.

### Final Recommendation

This audit does **not** rescue Dalgleish as a clean main-text full-vine win.

What it does show is more precise:

- the benchmark was too compressed before,
- the raw data are not neuron-poor, but the slice-wise trial budget is too small for large direct neuron sets,
- biologically better timing helps,
- richer short-latency temporal structure helps,
- `spks` is the right simple signal,
- and the best path forward is a small delayed/post-stim temporal representation, not a larger raw-neuron benchmark.

Paper-use recommendation after this audit:

- Dalgleish remains **supplement-worthy / control-worthy**
- not a clean main-text real-data win
- but now with a much clearer neuroscience-based explanation of what the dataset can and cannot support

## Representation-Formulation Viability Study: Proposed Plan

### Objective Of This Study

This next study is not asking which tiny benchmark variant has the best score.
It is asking a more basic question:

- what is the right **scientific object** for DVC on this dataset?

The three candidate objects are:

1. raw-neuron local subsets,
2. interpretable population-level summaries,
3. low-dimensional latent population states.

The goal is to determine which of these is actually viable on the current Dalgleish data, given what we already know:

- raw-neuron fitting becomes sample-limited quickly,
- better windowing and temporal structure help,
- but the best current neuron-level result is still too small and heterogeneous for a satisfying main-text result.

### What I Will Hold Fixed

These parts stay fixed and will not be reopened:

- author-aligned trial extraction,
- stable family restriction only,
- train-only winsorization and empirical-CDF mapping,
- same Gaussian / `1`-truncated vine / full-vine comparison,
- same repeated train/test splitting within `session x dose`,
- same `11`-session source dataset when feasible.

### Why These Three Families Are The Right Comparison

#### Family A: Raw-Neuron Local Subsets

Why this family matters:

- this is the closest to the current neuron-level benchmark,
- it preserves the interpretation of dependence among specific selected neurons,
- but it may be scientifically too local and statistically too undersampled.

The core question for Family A is:

- is raw-neuron DVC still viable at all here, if we keep the representation small but biologically sensible?

#### Family B: Interpretable Population Summaries

Why this family matters:

- the biological question in this dataset is arguably about **network recruitment**, not about a handful of individually selected neurons,
- population summaries can use information from many neurons while staying low-dimensional and interpretable,
- and they may better match the exogenous-intervention story in the paper.

The core question for Family B is:

- does DVC become more scientifically meaningful when the variables are population recruitment descriptors rather than neuron identities?

#### Family C: Low-Dimensional Latent Population States

Why this family matters:

- this is the cleanest way to use many neurons without trying to fit a direct high-dimensional neuron-level copula,
- it may be the statistically realistic formulation on this dataset,
- but it changes the interpretation from neuron interaction modeling to population-state modeling.

The core question for Family C is:

- even if raw-neuron DVC is not the right formulation here, is DVC viable on low-dimensional latent population-state trajectories?

### Proposed Signal And Window Backbone

Unless local inspection reveals a contradiction, I will use the current best-supported backbone:

- signal: `spks`
- response timing: skip the earliest `0.0–0.2 s`
- main windows:
  - delayed stim: `0.2–0.7 s`
  - early post: `0.7–1.4 s`

Reason:

- the previous audit already showed that immediate broad stim averaging is weak,
- `spks` was the strongest available simple signal,
- and delayed/post-stim windows were more biologically and statistically plausible.

### Exact Variants I Would Run

I will keep this compact.

#### Family A: Raw-Neuron Local Subsets

I would run exactly these `3` variants:

1. `A1_raw_4n_post_scalar`
   - `4` non-targeted responsive neurons
   - `1` scalar per neuron
   - feature meaning:
     - “selected neuron `i` mean `spks` from `0.7–1.4 s`”
   - total dimension: `4`

2. `A2_raw_4n_2bin`
   - `4` non-targeted responsive neurons
   - `2` bins per neuron
   - feature meaning:
     - “selected neuron `i` mean `spks` from `0.2–0.7 s`”
     - “selected neuron `i` mean `spks` from `0.7–1.4 s`”
   - total dimension: `8`

3. `A3_raw_8n_post_scalar`
   - `8` non-targeted responsive neurons
   - `1` scalar per neuron
   - feature meaning:
     - “selected neuron `i` mean `spks` from `0.7–1.4 s`”
   - total dimension: `8`

Why these three:

- they span the scientifically viable raw-neuron range suggested by the last audit,
- they compare scalar vs short temporal structure at matched modest dimensionality,
- and they test whether a somewhat larger raw-neuron subset is still viable without becoming obviously undersampled.

#### Family B: Interpretable Population Summaries

I would run exactly these `3` variants:

1. `B1_pop_post_4d`
   - all mapped targeted neurons and all non-targeted usable neurons contribute
   - variables:
     - targeted mean post response `0.7–1.4 s`
     - non-targeted mean post response `0.7–1.4 s`
     - non-targeted response standard deviation `0.7–1.4 s`
     - non-targeted upper-tail response `90th percentile`, `0.7–1.4 s`
   - total dimension: `4`

2. `B2_pop_temporal_6d`
   - all mapped targeted neurons and all non-targeted usable neurons contribute
   - variables:
     - targeted mean delayed response `0.2–0.7 s`
     - targeted mean early-post response `0.7–1.4 s`
     - non-targeted mean delayed response `0.2–0.7 s`
     - non-targeted mean early-post response `0.7–1.4 s`
     - non-targeted response standard deviation `0.2–0.7 s`
     - non-targeted response standard deviation `0.7–1.4 s`
   - total dimension: `6`

3. `B3_pop_recruitment_6d`
   - all mapped targeted neurons and all non-targeted usable neurons contribute
   - variables:
     - targeted mean early-post response `0.7–1.4 s`
     - non-targeted mean early-post response `0.7–1.4 s`
     - non-targeted upper-tail response `90th percentile`, `0.7–1.4 s`
     - non-targeted lower-tail response `10th percentile`, `0.7–1.4 s`
     - non-targeted delayed-to-post change: mean(`0.7–1.4 s`) minus mean(`0.2–0.7 s`)
     - targeted delayed-to-post change: mean(`0.7–1.4 s`) minus mean(`0.2–0.7 s`)
   - total dimension: `6`

Why these three:

- they are low-dimensional and neuroscience-interpretable,
- they use many neurons instead of a tiny selected subset,
- and they represent the recruitment story more directly than raw-neuron identities.

#### Family C: Low-Dimensional Latent Population States

I would run exactly these `3` variants:

1. `C1_latent_post_pca4`
   - source space:
     - all non-targeted usable neurons
     - post window `0.7–1.4 s`
   - variables:
     - first `4` PCA scores fit on train only
   - total dimension: `4`

2. `C2_latent_2bin_pca4`
   - source space:
     - all non-targeted usable neurons
     - delayed and post windows `0.2–0.7 s` and `0.7–1.4 s`
   - variables:
     - first `4` PCA scores fit on train only from the concatenated neuron x bin matrix
   - total dimension: `4`

3. `C3_latent_2bin_pca6`
   - source space:
     - all non-targeted usable neurons
     - delayed and post windows `0.2–0.7 s` and `0.7–1.4 s`
   - variables:
     - first `6` PCA scores fit on train only
   - total dimension: `6`

Why these three:

- they test whether low-dimensional population-state variables are the statistically realistic formulation here,
- they keep the latent model transparent,
- and they let us compare scalar population states vs temporally structured population states.

### How I Will Keep The Comparison Fair

To compare formulations fairly, I would use the following rules:

1. Same split policy across families
   - repeated train/test splits within `session x dose`
   - same random seeds for matched repeats

2. Same signal backbone where possible
   - `spks`
   - same delayed/post response windows

3. Train-only transforms everywhere
   - neuron selection for Family A fit on train only
   - any thresholds used for population summaries fit on train only if needed
   - PCA fit on train only and applied to test
   - ECDF mapping fit on train only and applied to test

4. Coverage reported explicitly
   - for each variant I will report:
     - number of usable sessions
     - number of usable `session x dose x repeat` slices
     - any failure or instability rate

5. Main comparison on the common slice intersection when useful
   - I will report native coverage for each variant,
   - and where needed I will also compare the families on the intersection of slices shared by the main viable variants, so we do not confuse formulation quality with different slice availability.

### What One Feature Means In Each Family

This will be written explicitly to a semantics table, but in plain language:

- Family A feature:
  - “this is one selected neuron’s response in a specified post-stim time bin”

- Family B feature:
  - “this is a trial-level population descriptor computed from many neurons, such as targeted mean response or non-targeted response spread”

- Family C feature:
  - “this is a latent population-state coordinate, for example PCA component `1`, derived from many neurons and fit on train only”

This matters because these families answer different scientific questions:

- Family A asks about local neuron-level dependence
- Family B asks about interpretable recruitment-summary dependence
- Family C asks about low-dimensional population-state dependence

### Outputs I Would Write In Stage 2

If approved, I would write:

- `dvc_ready/formulation_viability_summary.csv`
- `dvc_ready/formulation_metrics_table.csv`
- `dvc_ready/formulation_feature_semantics.csv`

And I would also mirror the main artifacts under:

- `results/stimulation_exp_benchmark/data/`

Likely files:

- `results/stimulation_exp_benchmark/data/formulation_viability_summary.csv`
- `results/stimulation_exp_benchmark/data/formulation_metrics_table.csv`
- `results/stimulation_exp_benchmark/data/formulation_feature_semantics.csv`
- `results/stimulation_exp_benchmark/data/formulation_viability_metadata.json`

Planned figures:

- `fig_formulation_family_comparison.png`
- `fig_formulation_best_raw_neuron.png`
- `fig_formulation_best_population_summary.png`
- `fig_formulation_best_latent_state.png`

### How I Would Decide Among The Three Families

For each variant and each family, I would summarize:

- Gaussian held-out NLL
- `1`-truncated vine held-out NLL
- full-vine held-out NLL
- delta vs Gaussian
- `TC_higher`
- proportion of slices where full vine beats Gaussian
- proportion of slices where full vine beats `1`-trunc
- stability / failure rate
- a brief interpretability note

Then I would classify each family as one of:

- `not_viable`
- `technically_viable_but_scientifically_weak`
- `scientifically_interpretable_but_not_competitive`
- `best_current_direction`

### What This Study Would Let Us Conclude

- If Family A remains best:
  - raw-neuron DVC is still viable enough to keep pursuing, even if the representation must stay small.

- If Family B is best:
  - the right scientific object here is not local neuron identities but interpretable population recruitment summaries.

- If Family C is best:
  - the statistically realistic formulation is low-dimensional population-state DVC, not neuron-level DVC.

- If none are good enough:
  - the dataset remains useful mainly as a supplement/control example rather than a main real-data showcase.

### Estimated Scope

This would still be a focused study, not a broad sweep.

Expected scope:

- `3` raw-neuron variants
- `3` population-summary variants
- `3` latent-state variants
- one compact comparison pass using the already validated pipeline pieces

Estimated runtime:

- similar to a medium-sized audit run
- larger than the last Stage 1 planning pass
- smaller than a broad new benchmark campaign

## Representation-Formulation Viability Study: Results

### Objective Of This Run

This run asked a formulation question rather than another local benchmark question:

- on the current Dalgleish dataset, is DVC most viable as:
  - a raw-neuron local-subset model,
  - an interpretable population-summary model,
  - or a low-dimensional latent population-state model?

### What Was Run

The study used the fixed validated pipeline:

- author-aligned trial extraction,
- stable family restriction only,
- train-only preprocessing and empirical-CDF mapping,
- repeated train/test splits within `session x dose`,
- and the same delayed/post-stim `spks` backbone where possible.

The tested variants were:

- Family A:
  - `A1_raw_4n_post_scalar`
  - `A2_raw_4n_2bin`
  - `A3_raw_8n_post_scalar` as a stretch test
- Family B:
  - `B1_pop_post_4d`
  - `B2_pop_temporal_6d`
  - `B3_pop_recruitment_6d`
- Family C:
  - `C1_latent_post_pca4`
  - `C2_latent_2bin_pca4`
  - `C3_latent_2bin_pca6`

### Coverage And Common-Slice Comparison

This turned out to be cleaner than expected.

All nine variants were viable on the same:

- `11` sessions
- `154` usable `session x dose x repeat` slices

So the mandatory common-slice comparison was not a small restrictive subset.
It was effectively the full viable benchmark.

This means the family comparison is **not** being driven by uneven coverage.

### Family A: Raw-Neuron Local Subsets

Raw-neuron results:

| variant | full delta vs Gaussian | prop full < Gaussian | prop full < 1-trunc | mean TC_higher |
|---|---:|---:|---:|---:|
| `A1_raw_4n_post_scalar` | `+0.045` | `0.571` | `0.571` | `+0.031` |
| `A2_raw_4n_2bin` | `+0.183` | `0.519` | `0.390` | `-0.171` |
| `A3_raw_8n_post_scalar` | `+0.341` | `0.708` | `0.448` | `-0.145` |

Interpretation:

- Raw-neuron DVC is still technically viable on this dataset.
- But the scientifically cleaner raw-neuron story remains weak.
- The smallest scalar raw-neuron setup (`A1`) is the only one with mildly positive average `TC_higher`, but the effect is tiny.
- The stronger raw-neuron delta-vs-Gaussian results (`A2`, `A3`) do **not** come with positive average higher-order gain.

Important stretch-test note:

- `A3_raw_8n_post_scalar` did **not** collapse in coverage, but it was still treated as a stretch test rather than the deciding raw-neuron benchmark.
- It improved full-vs-Gaussian, but its negative average `TC_higher` means it still does not give a satisfying higher-order raw-neuron story.

Bottom line for Family A:

- `technically_viable_but_scientifically_weak`

### Family B: Interpretable Population Summaries

Population-summary results:

| variant | full delta vs Gaussian | prop full < Gaussian | prop full < 1-trunc | mean TC_higher |
|---|---:|---:|---:|---:|
| `B1_pop_post_4d` | `-0.277` | `0.266` | `0.435` | `-0.125` |
| `B2_pop_temporal_6d` | `-0.581` | `0.201` | `0.481` | `-0.241` |
| `B3_pop_recruitment_6d` | `-0.252` | `0.292` | `0.506` | `-0.091` |

Interpretation:

- This family is the most neuroscience-interpretable.
- But numerically it is clearly worse than both Family A and Family C.
- The best population-summary variant was `B3_pop_recruitment_6d`, which is the most plausible recruitment-summary formulation of the three.
- Even so, full vine under this family was worse than Gaussian on average and only about neutral versus `1`-trunc.

Scientific reading:

- The population summaries are probably too aggregated.
- They may wash out the structured dependence differences that survive in the better raw and latent formulations.

Bottom line for Family B:

- `scientifically_interpretable_but_not_competitive`

### Family C: Low-Dimensional Latent Population States

Latent-state results:

| variant | full delta vs Gaussian | prop full < Gaussian | prop full < 1-trunc | mean TC_higher | retained PCA variance |
|---|---:|---:|---:|---:|---:|
| `C1_latent_post_pca4` | `+0.254` | `0.857` | `0.656` | `+0.063` | `0.284` |
| `C2_latent_2bin_pca4` | `+0.224` | `0.760` | `0.669` | `+0.068` | `0.277` |
| `C3_latent_2bin_pca6` | `+0.438` | `0.851` | `0.721` | `+0.198` | `0.384` |

Interpretation:

- This family is the most statistically successful on the current dataset.
- The best latent variant was `C3_latent_2bin_pca6`.
- It beat Gaussian strongly on average, beat `1`-trunc in about `72%` of slices, and had clearly positive average `TC_higher`.

PCA meaning:

- `C1` and `C2` retain about `28%` of the source variance.
- `C3` retains about `38%` of the source variance.

So the latent-state representation is not a nearly lossless compression of the population.
But it does retain a meaningful chunk of the structured trial-to-trial variation while staying low-dimensional enough to fit reliably.

Scientific reading:

- Family C is viable if we are willing to interpret the variables as **population-state coordinates**, not neuron-level interactions.
- This is a different scientific object from the original neuron-level DVC story.

Bottom line for Family C:

- `best_current_direction`

### Common-Slice Family Comparison

Because all families shared the same `154` viable slices, the common-slice comparison equals the native comparison for the representative variants:

- Family A representative:
  - `A2_raw_4n_2bin`
  - full delta vs Gaussian `+0.183`
  - mean `TC_higher = -0.171`
- Family B representative:
  - `B3_pop_recruitment_6d`
  - full delta vs Gaussian `-0.252`
  - mean `TC_higher = -0.091`
- Family C representative:
  - `C3_latent_2bin_pca6`
  - full delta vs Gaussian `+0.438`
  - mean `TC_higher = +0.198`

Interpretation:

- Family B was **not** numerically close to Family C.
- Therefore the interpretability tie-breaker in favor of Family B did **not** activate.
- Family C is genuinely better on the current benchmark, not just marginally better.

### What The Features Mean In Practice

The detailed semantics are written in:

- [formulation_feature_semantics.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/formulation_feature_semantics.csv)

In plain language:

- Family A features are:
  - specific selected neuron responses in delayed or early-post windows
- Family B features are:
  - interpretable population descriptors such as targeted mean response, non-targeted spread, and delayed-to-post recruitment change
- Family C features are:
  - PCA coordinates of non-targeted population activity fit on train only

### Final Family Classifications

- Family A:
  - `technically_viable_but_scientifically_weak`
- Family B:
  - `scientifically_interpretable_but_not_competitive`
- Family C:
  - `best_current_direction`

### Direct Answers To The Formulation Questions

1. **Is raw-neuron DVC viable enough on this dataset to keep pursuing?**
   Only weakly. It is technically workable at low dimension, but the neuron-level story remains too local and does not give a convincing higher-order result.

2. **Are interpretable population summaries a better scientific object?**
   They are a better scientific object in terms of interpretability, but not in terms of benchmark viability on this dataset. In this study they were too aggregated and not competitive.

3. **Are latent population-state variables the most realistic way to use DVC here?**
   Yes. On the current dataset, low-dimensional latent population-state variables are the most realistic statistically viable formulation.

4. **Which one should we carry forward, if any?**
   If we carry one forward, it should be **Family C**, with the explicit framing that this is a population-state DVC example rather than a neuron-level interaction example.

### Final Recommendation

This study changes the representation conclusion in an important way.

The current best direction on Dalgleish is **not**:

- tiny raw-neuron subsets
- and not interpretable population summaries

It is:

- low-dimensional latent population-state modeling

That said, this comes with a scientific tradeoff:

- Family C is the best current direction statistically,
- but it changes the interpretation away from direct neuron-level dependence.

So the most honest paper-level reading is:

- if the paper needs a real-data example of **DVC on low-dimensional population states**, Dalgleish can still be useful,
- but if the paper needs a clean neuron-level or directly interpretable recruitment-summary example, Dalgleish is still not the right main-text showcase.

Outputs generated in this pass:

- [formulation_viability_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/formulation_viability_summary.csv)
- [formulation_metrics_table.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/formulation_metrics_table.csv)
- [formulation_feature_semantics.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/formulation_feature_semantics.csv)
- [fig_formulation_family_comparison.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_formulation_family_comparison.png)
- [fig_formulation_best_raw_neuron.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_formulation_best_raw_neuron.png)
- [fig_formulation_best_population_summary.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_formulation_best_population_summary.png)
- [fig_formulation_best_latent_state.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_formulation_best_latent_state.png)

## Latent-State Full-Run Plan: Scientific Questions, Comparisons, And Interpretability

### What A Full Latent-State DVC Run Would Actually Compute

In plain language, a latent-state DVC run would do the following for each analysis slice:

- choose a **source space** of neurons:
  - non-targeted only,
  - targeted only,
  - or mixed targeted + non-targeted
- build a trial-by-neuron or trial-by-`(neuron x time-bin)` matrix from the `spks` signal
- fit PCA on the **training trials only** within that slice
- project both train and test trials into the retained low-dimensional latent coordinates
- apply the same train-only winsorization and train-only ECDF mapping used elsewhere
- fit Gaussian, `1`-truncated vine, and full vine on the train latent coordinates
- score all three models on the held-out latent coordinates from the same test trials

So “latent-state DVC” here means:

- DVC on train-only low-dimensional coordinates derived from many neurons
- not DVC on the neurons directly

Operationally, one latent variable means:

- a trial’s score on a train-fit PCA component summarizing a large pattern of population activity

The copula then models dependence among those latent coordinates across trials.

### What Matrix Goes Into PCA

For the current best latent formulation, the source matrix before PCA is:

- rows:
  - valid trials within a given `session x dose x repeat` slice
- columns:
  - all source-space neurons, optionally multiplied by a small number of response bins

Concretely, for the current best family backbone:

- signal:
  - `spks`
- windows:
  - delayed stim `0.2–0.7 s`
  - early post `0.7–1.4 s`
- source matrix for two-bin latent PCA:
  - `[delayed_bin(neuron_1), ..., delayed_bin(neuron_N), post_bin(neuron_1), ..., post_bin(neuron_N)]`

PCA would be fit:

- separately for each session and train/test split
- using **train trials only**

This avoids leakage because:

- the PCA basis is never fit on test trials
- the ECDF mapping is never fit on test trials
- the copula is never fit on test trials

### Why Non-Targeted Neurons Were Used So Far

The current rationale for non-targeted-only latent PCA is:

- to focus on recruited network activity rather than the most directly driven cells
- to reduce the chance that the latent space is dominated by an obvious stimulation-amplitude axis from targeted neurons alone
- and to ask whether the broader population response has non-Gaussian or higher-order dependence

That is a reasonable starting point, but it is not the only meaningful choice.

For the full latent-state run, the source space should be compared explicitly across:

- non-targeted only
- targeted only
- mixed targeted + non-targeted

### What Scientific Questions A Latent-State Run Can Answer

A latent-state run **can** answer questions like:

- does low-dimensional population activity show non-Gaussian dependence?
- does low-dimensional population activity show dependence beyond pairwise-only structure?
- does that latent dependence change with stimulation dose?
- does that latent dependence change over trial order within session?
- are the strongest effects more consistent in non-targeted, targeted, or mixed source spaces?
- are the dominant latent coordinates mostly global gain-like or more structured?

A latent-state run **cannot** directly answer questions like:

- which exact neurons form a higher-order dependence motif
- which triplets or quartets of neurons interact conditionally
- direct synaptic or circuit-mechanistic claims
- cell-type-specific mechanistic explanations

It is therefore a population-state analysis, not a direct neuron-level circuit analysis.

### What Time-Dependency Would Mean In Latent Space

In latent space, time-dependency would mean:

- the dependence among latent coordinates changes over trial order within session

The most interpretable compact version is:

- early / middle / late trial-order blocks within session
- evaluated at one or two eligible doses per session, prioritizing the strongest dose with enough trials

Evidence for meaningful temporal dependence would be:

- systematic changes in held-out full-vs-Gaussian gain across blocks
- and especially systematic changes in `TC_higher` across blocks
- that appear across more than a trivial number of sessions

This would not mean that the latent coordinates themselves merely drift in mean level.
It would mean the **dependence structure** among them changes over trial order.

### What Higher-Order / Non-Gaussian Dependence Would Mean In Latent Space

If full vine beats Gaussian in latent space, that would mean:

- the joint dependence among the retained latent coordinates is not well captured by an elliptical Gaussian copula alone

If full vine also beats `1`-trunc and `TC_higher` is positive, that would mean:

- dependence among latent population-state coordinates is not fully explained by pairwise-only structure
- and deeper conditional structure adds held-out predictive value

What we could honestly say from that:

- low-dimensional population-state activity contains higher-order or conditional dependence structure
- that structure may change with dose or trial order

What we could **not** honestly say from that alone:

- specific neurons or synapses form that higher-order interaction
- a specific circuit motif has been identified

### Can It Relate Back To Circuit Structure At All?

Yes, but only indirectly unless extra analyses are added.

The strongest indirect links we can plan are:

- whether top latent loadings are concentrated near stimulation targets
- whether mixed-space PCs place disproportionate weight on targeted versus non-targeted neurons
- whether certain PCs are dominated by delayed versus post-stim bins
- whether split-to-split stable PCs correspond to more spatially local versus global recruitment structure

These checks can support statements like:

- “the dominant latent mode appears more targeted-driven”
- “the dominant latent mode appears more distributed/recruited”
- “the latent dependence is carried by temporally later recruitment-like activity”

But they still do not justify a direct neuron-level mechanism claim.

### Can Behavior Be Incorporated?

Behavior is the main conditional piece in this plan.

What is available **now** in the validated builder:

- no trial-level behavior columns are currently exported in `trial_table.csv`

What is available in the local raw/session files:

- `BhvTraining ... VarFile ... .mat` files with fields like:
  - `laser_trials`
  - `pyb`
  - `stim`
  - `protocol`
- these appear to encode trial design / stimulation program information
- but the current builder does not yet convert them into trial-level response or reaction-time variables

What the authors’ repository documents:

- in the processed `DalgleishHausser2020_imaging_raw.mat` summary file, trial-level behavior variables exist as:
  - `tw_response`
  - `tw_rxn`
- that processed summary file is documented in the repo README but is **not** currently present under our local raw dataset root

So the behavior plan should be:

- conditional, not assumed

If trial-level behavior can be extracted reliably with modest effort from the local raw sidecars or paq behavior channels, the most compact behavior analysis would be:

- append one behavioral variable to the latent vector, preferably:
  - binary lick / response outcome
  - and optionally reaction time if reliable and sufficiently complete

This would answer:

- whether neural latent-state dependence jointly couples with behavioral response

If reliable trial-level behavior extraction is **not** available without opening a new extraction project, then behavior should be omitted from the first full latent-state run and explicitly documented as future work.

### Exact Comparisons To Run In The Full Latent-State Analysis

#### 1. Source-Space Comparison

This should be the first comparison.

I would run a compact source-space screen with the same latent formulation backbone:

- `non_targeted_2bin_pca4`
- `targeted_2bin_pca4`
- `mixed_2bin_pca4`

Each comparison answers:

- non-targeted only:
  - is higher-order structure strongest in recruited network activity?
- targeted only:
  - is the signal mostly driven by directly stimulated cells?
- mixed:
  - is the important dependence specifically in the interaction between targeted and recruited populations?

Then I would add rank sensitivity on the best one or two source spaces:

- `pca4`
- `pca6`

This keeps the run compact while still answering whether the latent result is robust to source-space choice and retained rank.

#### 2. Dose Analysis In Latent Space

For the chosen main latent formulation, I would compute:

- held-out Gaussian / `1`-trunc / full-vine NLL by dose
- full-vs-Gaussian by dose
- `TC_higher` by dose

I would show both:

- a pooled across-session dose summary
- and a session-level robustness view with faint session lines or points

This answers:

- whether latent-state higher-order dependence increases, decreases, or peaks at certain doses

#### 3. Time-Dependency Analysis In Latent Space

The compact dynamic view should be:

- early / middle / late blocks within session
- at the strongest eligible dose per session
- with a minimum block size chosen so train/test splits remain stable

The primary plotted metrics should be:

- full-vs-Gaussian held-out gain by block
- `TC_higher` by block

This answers:

- whether latent-state dependence changes over trial order in a way that is stronger than simple mean drift

#### 4. Interpretability Sanity Checks

This is mandatory.

I would include exactly these checks:

- variance explained per retained PC
- cumulative variance retained by the chosen latent rank
- loading stability across repeated splits
- whether PCs are more delayed-dominated or post-dominated in the two-bin source space
- for mixed source spaces:
  - targeted versus non-targeted loading enrichment
- for non-targeted or mixed source spaces:
  - whether large-loading neurons are spatially closer to stimulation targets than expected from the full usable pool

What these checks would let us say:

- whether the latent space is stable enough to interpret at all
- whether the PCs look like global gain modes versus more structured recruitment modes
- whether latent structure is more targeted-driven, more recruited-network-driven, or mixed
- whether the stronger latent-state DVC result is built on meaningful population structure rather than arbitrary compression

#### 5. Behavior Integration

Behavior should be planned as:

- one compact optional module, only if reliable trial-level behavior can be extracted without destabilizing the validated run

Preferred options in order:

1. joint latent-state + binary response outcome copula
2. outcome-conditioned comparison of latent dependence
3. joint latent-state + reaction-time analysis if reaction time is sufficiently complete and not too sparse

What these would answer:

- whether latent neural population-state dependence relates to behavioral report

What they would **not** answer:

- a detailed perceptual or decision-making mechanism claim

### Exact Outputs To Generate In Stage 2 If Approved

Core outputs:

- latent static metrics table
- latent dose summary table
- latent dynamic summary table
- latent interpretability table
- latent source-space comparison table
- latent feature / PC semantics table

Likely filenames:

- `results/stimulation_exp_benchmark/data/latent_state_source_space_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_state_dose_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_state_dynamic_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_state_interpretability.csv`
- `results/stimulation_exp_benchmark/data/latent_state_feature_semantics.csv`
- `results/stimulation_exp_benchmark/data/latent_state_behavior_summary.csv` if behavior is feasible

Figures:

- source-space comparison figure
- dose summary figure
- session-level robustness-by-dose figure
- dynamic block figure
- PCA variance explained / loading stability figure
- behavior figure only if behavior extraction is reliable

### Scientific Questions Each Comparison Would Answer

- Source-space comparison:
  - where the latent dependence signal lives biologically: targeted, recruited, or mixed
- Dose comparison:
  - whether latent higher-order dependence depends on intervention strength
- Dynamic comparison:
  - whether latent dependence changes over trial order within session
- Interpretability checks:
  - whether the latent coordinates are stable, structured, and biologically interpretable enough to discuss
- Behavior module:
  - whether latent neural dependence relates to the animal’s behavioral report

### Risks And Interpretability Limits

The main risks are:

- targeted-only latent spaces may be too small or too trivially stimulus-driven
- mixed spaces may look better numerically but become harder to interpret
- latent PCs may be statistically useful but only moderately stable across splits
- positive higher-order dependence in latent space is still a **population-state** result, not a neuron-level circuit result
- behavior may not be reliable enough from the currently validated raw-data path without one more extraction layer

The most important honesty constraint is:

- even a strong full latent-state DVC result would support a claim about **low-dimensional population-state dependence**
- not a direct claim about specific neuronal interaction motifs

## Latent-State Full Run: Results And Scientific Recommendation

### Objective Of This Run

This run asked whether the strongest scientifically viable use of the Dalgleish dataset is a **latent population-state DVC formulation** rather than a neuron-level formulation.

The focus was:

- first, a mandatory source-space comparison across non-targeted, targeted, and mixed latent spaces,
- then dose and within-session temporal analyses in the best latent formulation,
- plus compact interpretability checks on the PCA coordinates,
- and an explicit early decision on whether behavior could be included.

### Behavior Feasibility Check

Behavior was checked **before** the main latent run.

Conclusion:

- **behavior is not feasible without a new extraction layer**

Why:

- raw `BhvTraining ... VarFile ... .mat` sidecars are present,
- but the current validated builder does not export reliable trial-level behavior variables such as outcome or reaction time,
- and the processed summary file documented in the authors' repository, which contains trial-level behavior fields, is not present locally.

Practical consequence:

- the main latent-state run was done as a neural-only analysis,
- and a joint neural-behavior latent copula should be treated as future work rather than folded into this validated pass.

### Mandatory Source-Space Comparison

This was the first scientific comparison in the run.

At matched PCA rank `4`, the three source spaces gave:

- `non_targeted_2bin_pca4`
  - full vs Gaussian: `+0.222`
  - `TC_higher`: `+0.074`
- `targeted_2bin_pca4`
  - full vs Gaussian: `+0.128`
  - `TC_higher`: `+0.009`
- `mixed_2bin_pca4`
  - full vs Gaussian: `+0.237`
  - `TC_higher`: `+0.062`

Important detail:

- the common-slice comparison and the native comparison were identical here
- all three source-space variants used the same `154` successful slices across `11` sessions

Interpretation:

- targeted-only latent space is the weakest of the three
- so the earlier choice to avoid a targeted-only latent space was scientifically justified
- mixed targeted + non-targeted latent space is competitive and slightly best at PCA rank `4`
- but once rank sensitivity was added, `non_targeted_2bin_pca6` became the best overall latent formulation

This gives the main scientific answer to the source-space question:

- **non-targeted-only was a reasonable starting choice, not just an arbitrary simplification**
- the latent dependence signal is not strongest in targeted-only space
- mixed space is real and competitive
- but the best current overall formulation is still the recruited/non-targeted latent space

### Chosen Main Latent Variant

The best overall latent variant in this run was:

- `non_targeted_2bin_pca6`

This means:

- source space:
  - all usable **non-targeted** neurons in the slice
- signal:
  - `spks`
- time representation:
  - delayed stim bin `0.2–0.7 s`
  - early post bin `0.7–1.4 s`
- dimensionality:
  - `6` train-fit PCA coordinates

What one modeled variable means here:

- one trial's score on one PCA component fit on the delayed/post-stim activity of many non-targeted neurons

So the copula is modeling dependence among **latent population-state coordinates**, not among neurons directly.

### Static Latent-State Performance

For `non_targeted_2bin_pca6`, the static held-out summary across the full run was:

- mean Gaussian NLL: `-0.159`
- mean `1`-truncated vine NLL: `-0.398`
- mean full-vine NLL: `-0.609`
- mean full-vs-Gaussian gain: `+0.450`
- mean `TC_higher`: `+0.211`
- proportion of slices where full beats Gaussian: `0.870`
- proportion of slices where full beats `1`-trunc: `0.727`

Interpretation:

- latent-state DVC is clearly viable on this dataset in a statistical sense
- the full vine is not merely competitive with Gaussian in latent space; it is substantially better on average
- and the positive `TC_higher` means deeper conditional structure adds held-out value beyond the pairwise-only vine baseline

What we can honestly say from this:

- the low-dimensional recruited-population state exhibits non-Gaussian and higher-order dependence
- this is a statement about latent population-state structure
- it is **not** a direct neuron-level circuit interaction claim

### Dose Analysis In Latent Space

The pooled dose summaries for the chosen latent variant were favorable at every dose.

Pooled full-vs-Gaussian gains:

- dose `5`: `+0.495`
- dose `10`: `+0.498`
- dose `25`: `+0.469`
- dose `50`: `+0.344`
- dose `75`: `+0.493`
- dose `100`: `+0.393`
- dose `200`: `+0.458`

Pooled `TC_higher`:

- dose `5`: `+0.334`
- dose `10`: `+0.235`
- dose `25`: `+0.229`
- dose `50`: `+0.093`
- dose `75`: `+0.296`
- dose `100`: `+0.148`
- dose `200`: `+0.144`

Interpretation:

- the latent-state signal is not restricted to only one dose extreme
- full-vine gains are broadly positive across the dose range
- but the dose pattern is **not** a clean monotone dose-response
- the result is therefore stronger as evidence for a robust latent-state dependence effect than for a simple “higher dose always means stronger higher-order dependence” claim

### Time-Dependency In Latent Space

The dynamic analysis used:

- the strongest eligible dose per session
- early / middle / late trial-order blocks
- the same `non_targeted_2bin_pca6` latent formulation

Mean full-vs-Gaussian gain by block:

- early: `+0.018`
- middle: `-0.814`
- late: `+0.172`

Mean `TC_higher` by block:

- early: `-0.402`
- middle: `-0.360`
- late: `-0.151`

Interpretation:

- there is still a temporal axis in latent space
- but this dynamic result is **not cleanly favorable** to full-vine DVC
- the middle block in particular looks unstable or unfavorable
- and `TC_higher` is negative in all three coarse blocks

So the scientific answer here is:

- yes, we can test time-dependency in latent space
- but on the current dataset and current blockwise sample sizes, the dynamic latent result is too inconsistent to be a clean positive story

### Interpretability Sanity Checks

The chosen latent representation retained about `0.384` of train variance on average across the first `6` PCs.

Mean variance explained by PC:

- PC1: `0.085`
- PC2: `0.069`
- PC3: `0.063`
- PC4: `0.059`
- PC5: `0.055`
- PC6: `0.053`

Mean cumulative retained variance:

- through PC4: about `0.276`
- through PC6: about `0.384`

Loading stability across repeats:

- PC1: `0.623`
- PC2: `0.462`
- PC3: `0.401`
- PC4: `0.336`
- PC5: `0.301`
- PC6: `0.279`

Temporal weighting:

- post-bin weight fraction was about `0.50` for every retained PC

Interpretation:

- the first latent component is moderately stable and interpretable
- deeper PCs are progressively less stable across repeated splits
- the retained latent space is useful, but not so stable that we should make strong detailed claims about specific component identities
- the PCs do not look like purely post-dominated modes; delayed and early-post bins contribute roughly evenly on average

Scientifically, this supports a moderate claim:

- the latent-state result is real enough to analyze
- but the PCA basis is only moderately stable beyond the first one or two components
- so interpretability should stay at the level of **population-state structure**, not named mechanistic latent factors

### What This Full Latent-State Run Can And Cannot Support

This run **can** support claims like:

- low-dimensional population-state activity in this dataset has non-Gaussian dependence
- that dependence contains a higher-order / conditional component beyond pairwise-only structure
- the strongest current latent signal lives more in recruited/non-targeted or mixed population space than in targeted-only space
- dose-conditioned latent dependence is broadly positive and robust across sessions

This run does **not** support claims like:

- specific neuron triplets or motifs are interacting conditionally
- a synaptic or cell-type mechanism has been identified
- the temporal dependence story is already clean enough for a main dynamic result
- latent neural dependence has been linked to behavior in the current validated pipeline

### Final Paper Recommendation For The Latent-State Formulation

Final recommendation:

- **supplement/control only**

Why this is stronger than the neuron-level benchmark but still not clearly main-text:

- the static latent-state result is genuinely positive and is the best current use of the dataset
- the source-space comparison gives a clear and scientifically useful answer
- but the dose pattern is not cleanly monotone
- the dynamic latent result is inconsistent and not a clean DVC win
- and the PCA basis is only moderately stable beyond the top component

So the best honest use of Dalgleish after this run is:

- as a **latent population-state DVC example**
- not as a neuron-level interaction example
- and not yet as a clean flagship main-text real-data figure

### Bottom-Line Answers

1. **If we run the full latent-state formulation, what exactly is computed?**
   Train-only PCA coordinates from many neurons, followed by train-only ECDF mapping and copula fitting on those latent trial coordinates.

2. **Why was non-targeted-only used before, and was it the right choice?**
   It was used to focus on recruited-network activity rather than directly driven targeted responses. The new source-space comparison shows that was a reasonable choice: targeted-only is weakest, mixed is competitive, and non-targeted wins overall once latent rank is allowed to increase.

3. **Can we test dose-dependent effects in latent space?**
   Yes. In this run, latent full-vs-Gaussian gains were positive across all tested doses, but the pattern was not cleanly monotone.

4. **Can we test time-dependency in latent space?**
   Yes. But the early/middle/late block result here was too inconsistent to support a clean dynamic claim.

5. **If we see positive higher-order dependence in latent space, what can we say honestly?**
   We can say that low-dimensional population-state activity contains dependence beyond Gaussian and beyond pairwise-only structure. We cannot claim direct neuron-level circuit motifs from that alone.

6. **Should we carry this formulation forward?**
   Yes, if Dalgleish is kept at all. But it should be carried forward as a **latent population-state supplement/control analysis**, not as a direct neuron-level main-text showcase.

## Latent-State Follow-Up Plan: Scientific Claims, Next Analyses, And Figure Design

### Refined Scientific Claim

The follow-up goal is no longer to ask whether Dalgleish gives a clean neuron-level DVC showcase.
That answer is already no.

The refined question is:

- can the current latent-state result support a careful publication-ready claim that **low-dimensional population states derived from many neurons exhibit non-Gaussian and higher-order dependence**, and that this signal is strongest in recruited/non-targeted or mixed population activity rather than in targeted-only activity?

If the next analyses hold, the main scientific sentence would be:

- “In the Dalgleish photostimulation dataset, DVC gains are strongest in train-only latent population states derived from many neurons, especially in recruited/non-targeted activity, indicating non-Gaussian and higher-order dependence beyond Gaussian and pairwise-only baselines at the population-state level.”

This is still a **population-state** claim, not a neuron-level circuit-motif claim.

### What I Will Treat As Already Fixed

I will hold fixed:

- author-aligned trial extraction
- stable family restriction only: `ind`, `gaussian`, `student`, `clayton`
- train-only PCA fitting
- train-only winsorization and ECDF mapping
- the validated full latent backbone:
  - `spks`
  - delayed bin `0.2–0.7 s`
  - early-post bin `0.7–1.4 s`

I will also treat the following as already established:

- targeted-only latent space is weaker than non-targeted or mixed at matched rank `4`
- the best current latent variant is `non_targeted_2bin_pca6`
- behavior is not feasible on the current validated data path without a new extraction layer

### No New Contradiction, But One Important Analysis Risk

I do **not** currently see evidence of a missed bookkeeping bug in the latent-state results.

The one important follow-up risk is conceptual rather than a classic bug:

- the current dynamic analysis refits PCA separately inside each trial-order block
- so some block-to-block differences may reflect changes in the latent basis as well as changes in dependence

That means the dynamic follow-up should include:

- a primary dynamic analysis in the current blockwise train-only basis
- plus a sensitivity check using a common session-dose PCA basis fit before subdividing into blocks

If those disagree materially, the dynamic claim stays exploratory.

### Exact Planned Analyses

#### A. Static Latent-State Claim

Primary comparison set:

- `non_targeted_2bin_pca6`
- `mixed_2bin_pca6`
- `targeted_2bin_pca4`

Reason:

- these are the scientifically relevant source spaces
- they keep the current validated latent formulation backbone
- and they answer whether the strongest signal really lives in recruited/non-targeted activity

Main summaries to show:

- full-vs-Gaussian held-out NLL gap
- full-vs-`1`-trunc held-out NLL gap
- `TC_higher`
- proportion of session-level summaries where full beats Gaussian
- proportion of session-level summaries where full beats `1`-trunc

Primary visual:

- paired session points/lines on the common slice intersection
- one panel for full-vs-Gaussian
- one panel for `TC_higher`
- source spaces on the x-axis

Main scientific question answered:

- is the strongest latent DVC signal in recruited/non-targeted or mixed population activity, rather than targeted-only activity?

Paper strength:

- this is the strongest candidate **main-text** latent claim if the session-level effect remains clean

#### B. Dose-Dependent Latent-State Analysis

Primary question:

- not “is there a monotone dose trend?” by default
- but “is the latent-state DVC effect broadly positive across doses, and are some doses stronger than others?”

Planned summaries:

- pooled across-session mean by dose
- session-level by-dose summaries
- full-vs-Gaussian by dose
- `TC_higher` by dose
- optionally full-vs-`1`-trunc by dose if it helps interpretation

Primary inferential unit:

- session

Planned inference:

- bootstrap `95%` confidence intervals over session-level dose means
- paired sign-flip or permutation tests for whether the session-level mean full-vs-Gaussian is positive at each dose
- no formal monotone trend test unless the empirical curve actually looks monotone

Main scientific question answered:

- is the latent DVC gain broadly dose-robust, or does it localize to certain stimulation regimes?

Paper strength:

- likely strong enough for a figure panel if framed as broad positivity rather than a strict monotone dose law

#### C. Time-Dependency / Stimulation-History Analysis

Primary scientific question:

- is the messy current dynamic result telling us about adaptation/history in latent dependence, or just instability from low sample size and changing latent bases?

Planned primary dynamic analysis:

- strongest eligible dose per session
- rolling windows over trial order rather than only early/middle/late blocks
- window size chosen to keep held-out fits numerically stable
- step size small enough to see coarse evolution without producing a huge number of windows

Planned sensitivity comparison:

1. current blockwise train-only PCA within each window
2. common session-dose PCA basis fit before subdividing windows, then held fixed across windows

Primary summaries:

- full-vs-Gaussian over window center
- `TC_higher` over window center
- Gaussian held-out NLL over window center
- retained PCA variance over window center

Secondary diagnostic summaries:

- latent marginal variance by window
- window-level fit failure rate

Main scientific question answered:

- does latent dependence weaken in a middle/adaptation regime and recover later, or is the current block pattern mostly a basis/variance artifact?

Paper strength:

- exploratory unless the rolling-window and common-basis views agree

#### D. Family / Dependence-Type Analysis

This is a required next analysis.

Feasibility:

- the repository vine objects expose fitted pair-copula families through `cop.family`
- so a family-usage analysis is feasible with a modest instrumentation extension, not a new benchmark redesign

Planned family summaries:

- proportion of selected edges by family among the stable family set:
  - `ind`
  - `gaussian`
  - `student`
  - `clayton`
- summaries by tree level
- summaries by dose
- summaries by dynamic window/block for the chosen latent variant
- source-space comparison if feasible for the main source-space screen

Planned scientific compression of family results:

- independence
- Gaussian-like elliptical
- Student heavy-tailed elliptical
- Clayton asymmetric lower-tail

Main scientific question answered:

- is the latent DVC gain mostly coming from heavy-tailed elliptical dependence, asymmetric lower-tail dependence, or a mix?

Important limit:

- because the stable family set excludes `gumbel`, `joe`, and `frank`, the family-type interpretation is limited to the allowed stable set
- so “asymmetry” here mainly means **Clayton-like lower-tail asymmetry**, not every possible asymmetry class

Paper strength:

- likely supplement-level unless the selected-family pattern is very clear and stable
- but scientifically important because it sharpens what “non-Gaussian” means here

#### E. PC Interpretability Sanity Check

This remains mandatory.

Planned summaries:

- variance explained per retained PC
- cumulative variance explained
- loading stability across repeats
- delayed-bin vs post-bin loading balance
- targeted-vs-non-targeted enrichment in mixed space
- top-loading neuron proximity to stimulation targets
- session-level association between stronger DVC gain and:
  - PC1 stability
  - cumulative retained variance
  - post-vs-delayed balance
  - target proximity of top-loading neurons

Main scientific question answered:

- are stronger latent-state DVC effects associated with more stable, more distributed, or more target-proximal latent structure?

Interpretability limit:

- even if these checks are favorable, they justify only statements about the **character** of the latent population modes
- not direct neuron-level mechanism or motif claims

Paper strength:

- main-text supporting panel or supplement panel, depending on visual clarity

#### F. Neural-Behavioral Analysis

Decision for this follow-up:

- behavior is **not feasible with the current validated data path**
- behavior is feasible only as a separate future extraction project

Reason:

- no trial-level behavior variables are currently exported in the validated builder outputs
- and the documented processed summary file containing such variables is not present locally

Planned action in Stage 2:

- do not add a neural-behavioral analysis to the main latent follow-up run
- instead write an explicit one-paragraph limitation and future-work note

### Statistical Testing Plan

The publication-ready unit of replication will be:

- **session**, not split

Within each session I will first collapse repeated splits to session-level summaries, then run inference across sessions.

Primary static/source-space inference:

- summary statistic:
  - session-level mean full-vs-Gaussian
  - session-level mean `TC_higher`
- comparison:
  - non-targeted vs targeted
  - non-targeted vs mixed
- method:
  - paired sign-flip or permutation test on session-level differences
  - bootstrap `95%` confidence intervals on the paired mean difference

Dose inference:

- summary statistic:
  - session-level mean full-vs-Gaussian and `TC_higher` at each dose
- method:
  - bootstrap `95%` confidence intervals by dose
  - sign test or sign-flip test for positivity by dose

Dynamic inference:

- summary statistic:
  - session-level early/middle/late or rolling-window trend summaries
- method:
  - exploratory paired sign-flip comparisons such as early vs middle and middle vs late
- claim supported:
  - exploratory evidence for history dependence only if robust across both dynamic constructions

Family-selection inference:

- summary statistic:
  - session-level fraction of edges assigned to each family class
- method:
  - descriptive with bootstrap intervals
- claim supported:
  - descriptive characterization of dependence type, not a formal causal statement

### Publication-Ready Figure Plan

Proposed main publication-style figure:

- `fig_latent_publication_main.png`

Proposed layout:

1. Panel A: source-space comparison
   - paired session points/lines on common slices
   - y-axis: full-vs-Gaussian
   - x-axis: targeted, mixed, non-targeted

2. Panel B: higher-order gain by source space
   - paired session points/lines
   - y-axis: `TC_higher`
   - same x-axis ordering

3. Panel C: dose summary for the chosen main latent variant
   - per-session faint lines by dose
   - bold across-session mean with bootstrap CI ribbon or interval bars
   - y-axis: full-vs-Gaussian

4. Panel D: interpretability summary
   - PC variance explained and loading stability
   - likely as paired dot/line or point-range summaries rather than bars where possible

Planned supplement/debug-style figures:

- `fig_latent_publication_dynamic.png`
  - rolling-window latent dependence over trial order
- `fig_latent_publication_family_usage.png`
  - family usage by tree level and/or dose
- `fig_latent_publication_interpretability_extra.png`
  - target proximity and delayed-vs-post balance diagnostics

Visual defaults:

- prefer paired session points/lines
- prefer point-ranges or bootstrap intervals over bars
- use 1:1 scatter only if a direct full-vs-trunc or full-vs-Gaussian comparison is visually clearer than session lines

### README / Reproducibility Plan

Stage 2 should add a README section such as:

- `Dalgleish Latent-State Analysis`

That section should include:

- prerequisites:
  - repository environment
  - note that local runs here used `conda activate dvc`
- exact command for the current latent full run
- exact command for the new follow-up analysis script
- output directories:
  - `results/stimulation_exp_benchmark/data/`
  - `results/stimulation_exp_benchmark/plots/`
  - mirrored key CSVs in `dvc_ready/`
- a short description of:
  - source-space summary
  - dose summary
  - dynamic summary
  - family-selection summary
  - PC interpretability summary
  - publication figures

Planned follow-up script name:

- `scripts/run_dalgleish_latent_followup_analysis.py`

Planned key Stage 2 outputs:

- `results/stimulation_exp_benchmark/data/latent_followup_static_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_followup_dose_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_followup_dynamic_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_followup_family_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_followup_pc_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_followup_stats_summary.csv`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_main.png`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_dynamic.png`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_family_usage.png`

### Which Parts Are Strong Enough For The Paper

Strongest current paper candidates if they hold after the follow-up:

- static latent full-vs-Gaussian gain
- positive latent `TC_higher`
- source-space comparison showing targeted-only is weaker than non-targeted or mixed

Likely paper-supporting but not headline claims:

- dose effect is broadly positive across doses
- selected families are not purely Gaussian and include heavy-tailed or asymmetric components
- PCs are moderately stable and partly interpretable

Likely exploratory only:

- dynamic/history dependence
- any window-by-window family-switch interpretation

### What Stage 2 Will Try To Decide

After the follow-up, the main decisions should be:

1. Is the latent-state static claim strong enough for a main-text figure, or still better as supplement/control?
2. Is the strongest latent signal specifically recruited/non-targeted, mixed, or both?
3. Is the non-Gaussian gain mostly heavy-tailed elliptical, lower-tail asymmetric, or a mixed profile within the stable family set?
4. Is the dynamic signal interpretable enough to mention, or should it remain a secondary exploratory note?

## Latent-State Follow-Up Analysis: Results, Interpretation, And Paper Decision

### Objective Of This Follow-Up

This pass was designed to turn the current latent-state result into a publication-ready analysis, with the strongest priority on the **static source-space comparison**.

The primary question was:

- does the latent DVC signal really concentrate in recruited/non-targeted or mixed population activity, rather than in targeted-only space?

Secondary questions were:

- whether the latent effect is broadly dose-positive,
- what kind of dependence the selected full-vine families imply,
- whether stronger latent gains are associated with interpretable latent structure,
- and whether the dynamic result is strong enough to include beyond an exploratory note.

### What Was Run

The follow-up preserved the validated latent backbone:

- stable family set only: `ind`, `gaussian`, `student`, `clayton`
- train-only PCA
- train-only winsorization and ECDF mapping
- `spks` signal
- delayed/post bins `0.2–0.7 s` and `0.7–1.4 s`

The follow-up script then ran:

- a narrow static rerun for:
  - `targeted_2bin_pca4`
  - `mixed_2bin_pca6`
  - `non_targeted_2bin_pca6`
- session-level inference on the common slice intersection
- a pooled and session-level dose summary for the chosen main latent variant
- a family-selection summary using the fitted full-vine pair-copula families
- an exploratory rolling-window dynamic analysis with:
  - within-window train-only PCA
  - common-basis descriptive sensitivity
- a PC interpretability pass using the existing latent interpretability outputs
- and a README update with exact reproduction commands

### Static Source-Space Comparison: Primary Result

This is the strongest result from the follow-up and the main figure target.

Session-level mean full-vs-Gaussian:

- non-targeted latent space:
  - `+0.452`
  - bootstrap `95%` CI: `[+0.342, +0.546]`
  - one-sided sign-flip `p = 0.00049`
- mixed latent space:
  - `+0.406`
  - bootstrap `95%` CI: `[+0.278, +0.516]`
  - one-sided sign-flip `p = 0.00098`
- targeted-only latent space:
  - `+0.120`
  - bootstrap `95%` CI: `[+0.074, +0.161]`
  - one-sided sign-flip `p = 0.00146`

Session-level mean `TC_higher`:

- non-targeted latent space:
  - `+0.210`
  - bootstrap `95%` CI: `[+0.095, +0.315]`
  - one-sided sign-flip `p = 0.00488`
- mixed latent space:
  - `+0.160`
  - bootstrap `95%` CI: `[+0.069, +0.250]`
  - one-sided sign-flip `p = 0.00586`
- targeted-only latent space:
  - `+0.001`
  - bootstrap `95%` CI: `[-0.051, +0.045]`
  - one-sided sign-flip `p = 0.493`

Paired session-level differences:

- non-targeted minus targeted, full-vs-Gaussian:
  - `+0.332`
  - bootstrap `95%` CI: `[+0.235, +0.425]`
  - paired sign-flip `p = 0.00098`
- non-targeted minus targeted, `TC_higher`:
  - `+0.209`
  - bootstrap `95%` CI: `[+0.091, +0.325]`
  - paired sign-flip `p = 0.00879`
- non-targeted minus mixed:
  - not clearly different on either full-vs-Gaussian or `TC_higher`

Interpretation:

- the strongest latent DVC signal is clearly **not** in targeted-only space
- both non-targeted and mixed latent spaces are strong
- non-targeted is numerically best
- mixed is statistically close
- targeted-only is weaker and does not show a convincing higher-order gain

This is the central scientific result:

- the strongest DVC signal on this dataset lives in **recruited/non-targeted or mixed population-state space**, not in a latent space dominated only by directly stimulated neurons

### Dose Analysis In Latent Space

For the chosen main latent variant `non_targeted_2bin_pca6`, full-vs-Gaussian was positive at every tested dose.

Pooled full-vs-Gaussian by dose:

- dose `5`: `+0.481`, `p = 0.00195`
- dose `10`: `+0.541`, `p = 0.00146`
- dose `25`: `+0.521`, `p = 0.00098`
- dose `50`: `+0.337`, `p = 0.01465`
- dose `75`: `+0.454`, `p = 0.00195`
- dose `100`: `+0.375`, `p = 0.00146`
- dose `200`: `+0.455`, `p = 0.00049`

Pooled `TC_higher` by dose:

- clearly positive at doses `5`, `10`, `25`, `75`, and `200`
- weaker and not clearly positive at doses `50` and `100`

Interpretation:

- the latent-state DVC effect is **broadly positive across doses**
- the data still do **not** support a clean monotone dose-response claim
- the right scientific statement is:
  - latent non-Gaussian and higher-order dependence is present across the stimulation range, with some heterogeneity by dose

This is strong supporting evidence, but not the main headline.

### Dynamic / Stimulation-History Follow-Up

This analysis remained explicitly secondary and exploratory.

The new dynamic run compared:

1. rolling windows with train-only PCA fit within each window
2. a common-basis descriptive sensitivity view

The agreement between those two views for the sign of full-vs-Gaussian was only:

- `0.509`

That is not high enough to treat the rolling-window latent result as a robust main-text claim.

Additional warning signs:

- the common-basis descriptive view produced very large and unstable full-vs-Gaussian excursions
- the within-window train-only view was much milder
- mean dynamic `TC_higher` remained negative in the exploratory windowed summaries

Interpretation:

- there may well be stimulation-history effects in latent space
- but the current dataset and current rolling-window sample sizes do **not** let us separate dependence change cleanly from basis drift / local instability

Final status of the dynamic result:

- **exploratory only**
- mentionable as a caveat or future direction
- not strong enough for the main figure

### Family / Dependence-Type Analysis

This analysis sharpened what “non-Gaussian” means in the current latent result.

For the main static latent variant, grouped full-vine family usage was:

- heavy-tailed elliptical (`student`):
  - about `46%`
- lower-tail asymmetric (`clayton`):
  - about `29%`
- independence:
  - about `24%`
- Gaussian-like elliptical (`gaussian`):
  - about `1%`

By tree level, the pattern stayed similar:

- `student` remained the single largest family class on every shown tree level
- `clayton` remained the second largest class
- pure Gaussian families were rare

By dose, the same broad pattern held:

- `student` typically around the low- to mid-`40%` range
- `clayton` around the high-`20%` range
- Gaussian families consistently rare

Interpretation:

- the latent-state DVC gain is **not** just a small deviation from a Gaussian copula
- it is driven mainly by:
  - heavy-tailed elliptical structure
  - plus a meaningful lower-tail asymmetric component

This is scientifically useful because it lets us say:

- the latent dependence is non-Gaussian in a specific way
- not just “full vine wins somehow”

This still remains a supporting result rather than the main figure headline.

### PC Interpretability

The latent PCs remain only moderately interpretable, but the picture is useful.

Variance explained per retained PC:

- PC1: `0.085`
- PC2: `0.069`
- PC3: `0.063`
- PC4: `0.059`
- PC5: `0.055`
- PC6: `0.053`

Loading stability:

- PC1: `0.623`
- PC2: `0.462`
- PC3: `0.401`
- PC4: `0.336`
- PC5: `0.301`
- PC6: `0.279`

The strongest follow-up association was:

- full-vs-Gaussian vs PC1 loading stability:
  - rank correlation about `+0.709`

Interpretation:

- stronger latent DVC gains tend to occur in sessions with a **more stable leading latent mode**

Important limitation:

- the planned target-proximity association could **not** be estimated directly for the winning non-targeted latent space
- by construction, that source space contains no targeted neurons, so the current proximity summary is undefined there
- that question is only addressable indirectly through the mixed-space interpretability summaries

So the biological interpretation we can justify is:

- stronger latent DVC gains are associated with **more stable latent recruited-population modes**
- but we still cannot make a clean direct claim that the winning non-targeted latent effect is specifically more target-proximal or more spatially distributed

### Behavior

Behavior remains unchanged from the full latent run:

- **not feasible with the current validated data path**

This follow-up did not reopen behavior extraction.
That remains a separate future project.

### Publication Figure Decision

The current best publication-style figure target is:

- [fig_latent_publication_main.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_main.png)

Why this is the best figure:

- Panel A and Panel B directly show the main scientific point:
  - targeted-only is weaker
  - non-targeted and mixed are stronger
  - and higher-order gain is concentrated outside targeted-only space
- Panel C shows the effect is broadly dose-positive rather than a fragile one-dose artifact
- Panel D keeps the interpretation honest by showing moderate latent stability rather than overclaiming

The dynamic and family-usage figures are useful supporting material, but not better main figures.

### Final Paper Decision

Final recommendation after this follow-up:

- **main-text ready**

That recommendation is now based on the **static source-space latent result**, not on the dynamic result.

Why this is now strong enough:

- the static source-space comparison is clean and session-level robust
- non-targeted and mixed spaces both show strong positive full-vs-Gaussian gains
- non-targeted shows clearly positive `TC_higher`
- targeted-only is significantly weaker than non-targeted on both full-vs-Gaussian and `TC_higher`
- the latent effect is broadly positive across doses
- and the selected families show a coherent non-Gaussian profile dominated by `student` plus `clayton`, not near-Gaussian fits

What this main-text claim should still **not** say:

- not neuron-level interaction motifs
- not direct circuit mechanism
- not a clean dynamic-history DVC win

The honest main-text claim is:

- Dalgleish provides a strong **latent population-state** real-data example for DVC
- especially for the claim that recruited/non-targeted population states exhibit non-Gaussian and higher-order dependence beyond Gaussian and pairwise-only baselines

### Outputs Generated

This follow-up wrote:

- [latent_followup_static_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_followup_static_summary.csv)
- [latent_followup_dose_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_followup_dose_summary.csv)
- [latent_followup_dynamic_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_followup_dynamic_summary.csv)
- [latent_followup_family_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_followup_family_summary.csv)
- [latent_followup_pc_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_followup_pc_summary.csv)
- [latent_followup_stats_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_followup_stats_summary.csv)
- [fig_latent_publication_main.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_main.png)
- [fig_latent_publication_dynamic.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_dynamic.png)
- [fig_latent_publication_family_usage.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_family_usage.png)
- [fig_latent_publication_interpretability_extra.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_interpretability_extra.png)

## Latent-State Publication Follow-Up Plan: Control, Source Space, Dynamic Families, And Final Figure

### Refined Scientific Claim

The publication-facing claim is now narrower and more precise:

- in the Dalgleish dataset, DVC gains are strongest in **latent population states** rather than raw neuron identities
- the clearest signal is in **non-targeted and mixed** latent population space, not targeted-only space
- and the gain reflects both:
  - flexible pairwise non-Gaussian structure
  - and an additional higher-order / conditional component

The claim we are trying to sharpen in this final publication pass is:

- stimulation appears to amplify or reshape latent non-Gaussian dependence relative to catch/control activity, and the strongest robust signal is in recruited/non-targeted or mixed population states

This remains a **population-state** claim, not a neuron-level mechanism claim.

### What Will Stay Fixed

I will keep fixed:

- author-aligned trial extraction
- stable family set only: `ind`, `gaussian`, `student`, `clayton`
- train-only PCA
- train-only winsorization and ECDF mapping
- latent backbone:
  - `spks`
  - delayed bin `0.2–0.7 s`
  - early-post bin `0.7–1.4 s`

I will also treat as fixed:

- the current static source-space latent result
- behavior not being feasible on the validated data path
- the dynamic result staying exploratory unless stronger agreement emerges

### One Important Constraint For The Control Analysis

Catch / dose `0` trials are present in all sessions, but only at about:

- `2` to `8` trials per session
- `60` catch trials total

This means:

- catch was excluded from the current latent benchmark by the validated within-session slice-size rules
- so a direct “same benchmark, but with dose 0” comparison is **not** currently feasible at the same latent rank and same session-level split design

Therefore the control analysis in the publication follow-up should be framed as:

- a **reduced-rank, exploratory control-vs-stim comparison**
- not the primary inference engine of the paper

If it works cleanly, it strengthens the story.
If it remains too sparse, the static source-space result still stands on its own.

### Exact Planned Analyses

#### A. Main Figure Redesign: Static Decomposition First

This should be the center of the final figure.

Primary comparison:

- `targeted_2bin_pca4`
- `mixed_2bin_pca6`
- `non_targeted_2bin_pca6`

Primary quantities:

1. pairwise flexible / low-level non-Gaussian gain:
   - `NLL(Gaussian) - NLL(1-trunc)`
2. higher-order / conditional gain:
   - `NLL(1-trunc) - NLL(full)`
   - this is `TC_higher`

Scientific meaning:

- Gaussian to `1`-trunc isolates the gain from allowing **pairwise flexible non-Gaussian structure**
- `1`-trunc to full isolates the gain from **deeper conditional / higher-order structure**

Planned visual:

- paired session point-range plots, not bars
- x-axis:
  - targeted
  - mixed
  - non-targeted
- y-axis:
  - one panel for Gaussian-to-`1`-trunc gain
  - one panel for `TC_higher`

Main scientific question answered:

- is the latent DVC gain mainly pairwise-flexible, higher-order, or both?
- and where does that signal live across source spaces?

Main figure role:

- **main figure**

#### B. Control / Catch Versus Stimulated Analysis

This is required, but should be planned honestly around the sparse catch condition.

Planned strategy:

1. first, verify whether a reduced-rank latent control comparison is feasible at all
   - likely `pca2` or `pca3`
   - for non-targeted and mixed source spaces only
2. if feasible, compare:
   - catch / dose `0`
   - stimulated trials pooled across nonzero doses
3. compute:
   - full-vs-Gaussian
   - Gaussian-to-`1`-trunc gain
   - `TC_higher`
   - grouped family mix

Because catch is sparse, this analysis should likely use:

- session as a blocking label where possible
- but be labeled **exploratory/supporting** unless enough sessions pass a minimal reduced-rank control benchmark

Scientific questions answered:

- is latent non-Gaussian dependence already present in catch trials?
- is stimulation mainly amplifying the magnitude of the effect?
- or does stimulation change the **type** of dependence as well?

If catch remains too sparse even after reduced rank:

- report that explicitly
- use control-vs-stim family and score comparisons only as descriptive pooled summaries

Main figure role:

- likely one supporting panel or supplement panel
- include in the main figure only if the control contrast is clean enough

#### C. Source-Space Comparison

This remains central and should stay in the main figure.

Compare:

- targeted-only
- mixed targeted + non-targeted
- non-targeted

Primary summaries:

- Gaussian-to-`1`-trunc gain
- `TC_higher`
- and, secondarily, full-vs-Gaussian

Scientific sentence supported if the result holds:

- the strongest latent nonlinear and higher-order dependence is not in targeted-only space, but in recruited/non-targeted or mixed population-state space

Main figure role:

- **main figure**

#### D. Dynamic / Stimulation-History Analysis

This remains important, but should stay secondary unless it becomes much cleaner.

Planned dynamic comparison:

1. early / middle / late block summaries for the chosen main latent variant
2. grouped family usage by early / middle / late block
3. same two basis views as before:
   - window/blockwise train-only PCA
   - common session-dose PCA basis sensitivity

Primary scientific questions:

- does the middle regime become more independent?
- more Gaussian-like?
- less heavy-tailed?
- less lower-tail-asymmetric?
- or does it remain too unstable to interpret?

What would count as evidence that stimulation history weakens or simplifies nonlinear dependence:

- lower Gaussian-to-`1`-trunc gain in middle than early
- lower `TC_higher` in middle than early
- a family shift toward:
  - more independence
  - or more Gaussian-like elliptical
  - and less student / less clayton
- agreement between the blockwise and common-basis views

If the two dynamic constructions still disagree clearly:

- keep the dynamic result **supplement/exploratory only**

Main figure role:

- supplement unless much cleaner than current results

#### E. Family / Dependence-Type Analysis

This is required and should now be publication-facing.

Report both:

- raw stable families:
  - `ind`
  - `gaussian`
  - `student`
  - `clayton`
- grouped publication-facing classes:
  - independence
  - Gaussian-like elliptical
  - heavy-tailed elliptical
  - lower-tail asymmetric

Planned family summaries:

- overall static latent result
- by source space
- by dose
- by early / middle / late block
- by control vs stimulated if the control analysis is feasible

Scientific questions answered:

- is the static latent gain driven mainly by heavy tails?
- by lower-tail asymmetry?
- does stimulation increase heavy-tailed or asymmetric structure relative to catch?
- does the middle block simplify toward more independence / more Gaussian-like structure?

Main figure role:

- likely a compact supporting panel if the control contrast is clean
- otherwise supplement

#### F. PC Interpretability

This remains mandatory, but more focused.

Primary biological question:

- are stronger latent DVC gains associated with more stable recruited-population modes, more target-proximal modes, or more distributed modes?

Planned summaries:

- variance explained
- loading stability
- delayed-vs-post weighting
- targeted-vs-non-targeted enrichment in mixed space
- target-proximity diagnostics where they are meaningful

Important interpretation rule:

- for the winning non-targeted latent space, direct target-proximity diagnostics are limited
- so the distributed-versus-target-proximal question may need to be addressed indirectly through:
  - mixed-space PCs
  - and source-space contrasts

Scientific question answered:

- does the strongest latent DVC signal track stable recruited population modes, and is there any evidence that those modes are more distributed or more target-linked?

Main figure role:

- one compact panel in the main figure
- extra diagnostics in supplement

### Statistical Testing / Uncertainty Plan

Replication unit:

- **session**

Primary statistics:

- session-level mean Gaussian-to-`1`-trunc gain
- session-level mean `TC_higher`
- session-level mean full-vs-Gaussian

Main inferential methods:

- paired sign-flip / permutation tests for source-space contrasts
- bootstrap `95%` confidence intervals for session-level means and mean paired differences
- paired session points/lines in figures

For control-vs-stim:

- if enough session-level reduced-rank comparisons are feasible:
  - use the same session-level bootstrap + paired sign-flip approach
- if not:
  - report descriptive pooled effect sizes with explicit caution that inference is limited by sparse catch trials

For dynamic early/middle/late:

- use session-level paired sign-flip comparisons
- but only support an interpretive claim if the blockwise and common-basis results agree directionally

For family usage:

- descriptive session-level fractions with bootstrap intervals
- no strong causal claims from family frequencies alone

### Main Figure Versus Supplement

#### Main Figure Candidate Panels

Proposed main figure:

- `fig_latent_publication_final.png`

Likely panel layout:

1. **Panel A: Source-space decomposition**
   - targeted vs mixed vs non-targeted
   - paired session points/lines
   - Gaussian-to-`1`-trunc gain

2. **Panel B: Source-space higher-order gain**
   - targeted vs mixed vs non-targeted
   - paired session points/lines
   - `TC_higher`

3. **Panel C: Control vs stimulated**
   - if feasible at reduced rank
   - otherwise replace with by-dose broad-positivity panel
   - show full-vs-Gaussian and/or grouped family mix

4. **Panel D: Dependence-type summary**
   - grouped family composition for the main latent variant
   - static overall or stim-vs-control if feasible

5. **Panel E: PC interpretability**
   - variance explained + stability
   - compact and honest

#### Supplement / Exploratory Panels

Keep these out of the main figure unless unexpectedly strong:

- rolling-window dynamic diagnostics
- common-basis versus blockwise dynamic comparison
- full early/middle/late family breakdowns
- extra PC enrichment and proximity panels
- extra by-dose family breakdowns

### README / Reproducibility Plan

The final publication pass should add a dedicated script, likely:

- `scripts/run_dalgleish_latent_publication_analysis.py`

The README update should include:

- exact environment note:
  - `conda activate dvc`
- exact command for the new publication analysis script
- output file locations under:
  - `results/stimulation_exp_benchmark/data/`
  - `results/stimulation_exp_benchmark/plots/`
  - mirrored key CSVs in `dvc_ready/`
- a short mapping from script to outputs:
  - publication static/source-space summary
  - control-vs-stim summary
  - family summary
  - dynamic block summary
  - PC summary
  - final publication figure

Likely new Stage 2 outputs:

- `results/stimulation_exp_benchmark/data/latent_publication_static_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_control_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_family_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_dynamic_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_pc_summary.csv`
- `results/stimulation_exp_benchmark/data/latent_publication_stats_summary.csv`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_final.png`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_dynamic_supplement.png`
- `results/stimulation_exp_benchmark/plots/fig_latent_publication_family_supplement.png`

### What This Publication Follow-Up Would Decide

After this pass, the main decisions should be:

1. Is the key publication panel the source-space decomposition into pairwise-flexible vs higher-order gain?
2. Can catch/control be included honestly, and if so, is the effect amplified or qualitatively changed by stimulation?
3. Does the family mix show that stimulation-related latent dependence is mainly heavy-tailed, lower-tail-asymmetric, or both?
4. Does the dynamic middle regime simplify toward more independence / more Gaussian-like structure, or remain too unstable to interpret?
5. What is the cleanest final main figure layout for the paper?

## Latent-State Publication Follow-Up: Results, Figure Set, And Final Recommendation

### Objective Of This Pass

This pass turned the current latent-state result into a publication-ready analysis with the updated figure priorities:

- **Panel A:** full vine versus all main baselines
- **Panel B:** decomposition into low-level / pairwise-flexible gain and higher-order / conditional gain
- **Panel C:** targeted vs mixed vs non-targeted source spaces
- **Panel D:** dose by default, with catch/control included only if a reduced-rank control comparison was clean enough

The benchmark backbone was held fixed:

- author-aligned trial extraction
- stable family set only: `ind`, `gaussian`, `student`, `clayton`
- train-only PCA
- train-only winsorization and ECDF mapping
- latent backbone: `spks`, delayed `0.2–0.7 s`, early-post `0.7–1.4 s`

### Outputs Generated

Code:

- [run_dalgleish_latent_publication_analysis.py](/Users/alessandro/Documents/github/DVC/scripts/run_dalgleish_latent_publication_analysis.py)
- [README.md](/Users/alessandro/Documents/github/DVC/README.md)

Data outputs:

- [latent_publication_static_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_static_summary.csv)
- [latent_publication_control_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_control_summary.csv)
- [latent_publication_family_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_family_summary.csv)
- [latent_publication_dynamic_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_dynamic_summary.csv)
- [latent_publication_pc_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_pc_summary.csv)
- [latent_publication_stats_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_stats_summary.csv)
- [latent_publication_baseline_feasibility.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_baseline_feasibility.csv)
- [latent_publication_metadata.json](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_metadata.json)

Figures:

- [fig_latent_publication_final.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_final.png)
- [fig_latent_publication_dynamic_supplement.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_dynamic_supplement.png)
- [fig_latent_publication_family_supplement.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_family_supplement.png)

The key CSVs and metadata were also mirrored into `dvc_ready/`.

### Panel A: Full Vine Versus All Main Baselines

The publication pass explicitly screened:

- Gaussian copula
- `1`-truncated vine
- Graphical Lasso Gaussian copula
- TVGL
- Gaussian SSM

What actually ran cleanly at the session level:

- Gaussian copula: yes
- `1`-truncated vine: yes
- Graphical Lasso: yes
- Gaussian SSM: yes
- TVGL: attempted, but **no usable session-level latent-static output**

This is now documented explicitly in:

- [latent_publication_baseline_feasibility.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_baseline_feasibility.csv)

Session-level mean `baseline NLL - full-vine NLL` for the main non-targeted latent variant:

- full vine vs Graphical Lasso: `+0.483`, bootstrap CI `[+0.363, +0.588]`, `p = 0.00049`
- full vine vs Gaussian SSM: `+0.487`, bootstrap CI `[+0.373, +0.595]`, `p = 0.00049`
- full vine vs Gaussian copula: `+0.442`, bootstrap CI `[+0.330, +0.532]`, `p = 0.00098`
- full vine vs `1`-trunc vine: `+0.206`, bootstrap CI `[+0.093, +0.307]`, `p = 0.00488`
- full vine vs TVGL: no valid session-level comparison in this latent-static setup

Interpretation:

- the static latent full vine is not only better than Gaussian copula and `1`-trunc;
- it is also better than the additional Gaussian-structured baselines that were runnable in the repository path on latent coordinates;
- and the remaining missing baseline is documented as a feasibility limit, not silently dropped.

### Panel B: Low-Level Versus Higher-Order Gain

For the main non-targeted latent variant:

- low-level / pairwise-flexible gain `Gaussian -> 1-trunc`: mean `+0.236`, bootstrap CI `[+0.178, +0.297]`, `p = 0.00049`
- higher-order / conditional gain `1-trunc -> full = TC_higher`: mean `+0.206`, bootstrap CI `[+0.089, +0.312]`, `p = 0.00488`

Interpretation:

- both parts of the decomposition are positive;
- so the latent gain is not only “non-Gaussian in a pairwise way”;
- there is also a significant higher-order / conditional component on top of the flexible pairwise gain.

### Panel C: Biological Source-Space Comparison

Session-level means on the common slice intersection:

- targeted latent space: `full_vs_gaussian = +0.119`, `gaussian_to_trunc = +0.117`, `TC_higher = +0.002`
- mixed latent space: `full_vs_gaussian = +0.420`, `gaussian_to_trunc = +0.247`, `TC_higher = +0.173`
- non-targeted latent space: `full_vs_gaussian = +0.438`, `gaussian_to_trunc = +0.235`, `TC_higher = +0.203`

Session-level paired inference:

- non-targeted minus targeted `full_vs_gaussian`: `+0.325`, bootstrap CI `[+0.233, +0.414]`, `p = 0.00098`
- non-targeted minus targeted `TC_higher`: `+0.208`, bootstrap CI `[+0.100, +0.309]`, `p = 0.00586`
- non-targeted minus mixed `full_vs_gaussian`: `+0.023`, CI crossing `0`, `p = 0.641`
- non-targeted minus mixed `TC_higher`: `+0.030`, CI crossing `0`, `p = 0.424`

Interpretation:

- targeted-only latent space is clearly weaker;
- the strongest signal lives in recruited/non-targeted and mixed latent spaces;
- and non-targeted and mixed are close enough that the clean scientific statement is:
  - latent DVC gains are strongest in recruited/non-targeted or mixed population-state space, not targeted-only space.

### Panel D: Dose, Not Catch/Control

Catch/control remained too sparse even after the reduced-rank screen:

- only `2` usable sessions for each reduced-rank control screen
- all control variants stayed `not_clean_enough`

This is recorded in:

- [latent_publication_control_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_control_summary.csv)

So the main figure uses **dose**, not catch/control.

Session-level mean `full_vs_gaussian` by dose for the main non-targeted latent variant:

- dose `5`: `+0.474`, CI `[+0.255, +0.694]`, `p = 0.00195`
- dose `10`: `+0.519`, CI `[+0.291, +0.784]`, `p = 0.00098`
- dose `25`: `+0.472`, CI `[+0.293, +0.670]`, `p = 0.00098`
- dose `50`: `+0.338`, CI `[+0.072, +0.567]`, `p = 0.01807`
- dose `75`: `+0.464`, CI `[+0.288, +0.634]`, `p = 0.00146`
- dose `100`: `+0.343`, CI `[+0.239, +0.446]`, `p = 0.00098`
- dose `200`: `+0.484`, CI `[+0.341, +0.673]`, `p = 0.00049`

Interpretation:

- the dose effect is broadly positive across all stimulated doses;
- it is not cleanly monotone;
- and there is not enough clean catch/control support to elevate control-vs-stim to the main figure.

### Family / Dependence-Type Analysis

The publication-facing grouped families are:

- independence
- Gaussian-like elliptical
- heavy-tailed elliptical
- lower-tail asymmetric

For the static source-space comparison:

- non-targeted: heavy-tailed elliptical `0.464`, lower-tail asymmetric `0.285`, independence `0.238`, Gaussian-like `0.013`
- mixed: heavy-tailed elliptical `0.470`, lower-tail asymmetric `0.278`, independence `0.242`, Gaussian-like `0.010`
- targeted: heavy-tailed elliptical `0.431`, lower-tail asymmetric `0.279`, independence `0.287`, Gaussian-like `0.003`

Interpretation:

- the static latent signal is not mostly Gaussian-like;
- it is dominated by `student` plus `clayton`, i.e. heavy-tailed elliptical plus lower-tail-asymmetric structure;
- and targeted-only space shifts toward more independence and less heavy-tailed structure than non-targeted or mixed space.

By dose in the main non-targeted latent variant:

- heavy-tailed elliptical remains the largest class at every dose
- lower-tail asymmetry remains the second main class
- Gaussian-like usage stays very small across doses

This supports the paper-facing statement that:

- stimulation-associated latent dependence is genuinely non-Gaussian;
- and within the stable family set it is best described as a mixture of heavy-tailed elliptical and lower-tail-asymmetric structure.

### Dynamic / History Analysis

This remains exploratory and stays out of the main figure.

Reasons:

- basis-sign agreement between blockwise-basis and common-basis `full_vs_gaussian` is only `0.517`
- window-train-basis dynamic mean `TC_higher` is negative (`-0.357`)
- common-basis dynamic mean `TC_higher` is even more unstable (`-1.619`)

The grouped family summaries also differ materially between basis constructions.

Interpretation:

- the current dynamic/history result is not robust enough to support a clean claim that stimulation history simplifies or weakens nonlinear latent dependence;
- it remains a supplement-level exploratory diagnostic only.

### PC Interpretability

The PCA interpretability summary is:

- [latent_publication_pc_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_pc_summary.csv)

Main findings:

- retained PCs remain only moderately stable, with PC1 stability strongest and later PCs progressively weaker
- mean post-weight fraction is about `0.50` across PCs, so the main latent space is not obviously dominated by either delayed or post bins alone
- stronger non-targeted gains are positively associated with PC1 stability (`rho ≈ +0.755`)
- in mixed space, stronger gains are only weakly associated with targeted enrichment (`rho ≈ +0.045`)
- in mixed space, stronger gains are moderately associated with **more target-proximal** PC1 loadings (`rho ≈ -0.555`, where negative means more proximal)

Interpretation:

- the strongest latent DVC gains are associated most clearly with **stable recruited-population modes**;
- the source-space comparison still points to recruited/non-targeted or mixed space as the main locus of signal;
- and the mixed-space proximity diagnostic suggests some target-linked structure, but not enough to change the population-state interpretation into a neuron-level mechanism claim.

### Final Figure Recommendation

Main figure:

- [fig_latent_publication_final.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_final.png)

Panel content:

- **Panel A:** full vine versus all usable main baselines
- **Panel B:** Gaussian-to-`1`-trunc and `1`-trunc-to-full decomposition
- **Panel C:** targeted vs mixed vs non-targeted source spaces
- **Panel D:** by-dose robustness for the main non-targeted latent variant

Supplement:

- [fig_latent_publication_dynamic_supplement.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_dynamic_supplement.png)
- [fig_latent_publication_family_supplement.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_family_supplement.png)

### Final Paper Decision

Final recommendation after the publication follow-up:

- **main-text ready**

But with an explicit scope condition:

- this is a **latent population-state** real-data result
- not a neuron-level interaction or motif result

The strongest defensible paper claim is:

- in Dalgleish, DVC gains are strongest in low-dimensional latent population states derived from many neurons, especially in recruited/non-targeted or mixed population activity;
- these latent states show non-Gaussian dependence beyond Gaussian baselines;
- and they also retain a positive higher-order / conditional component beyond the `1`-truncated pairwise-only vine.

## Publication Figure Redesign And Explanatory Overview Plan

### Objective Of This Pass

This pass is not a new benchmark run.
It is a figure-refresh and explanation pass built on the **existing validated latent-state outputs**.

The goal is to:

- turn the current static latent-state result into a cleaner publication-facing figure set
- move secondary analyses out of the main figure
- and add one clear overview document for a new reader

### Existing Results To Reuse

This refresh should reuse the current validated outputs rather than recomputing the benchmark logic:

- [latent_publication_static_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_static_summary.csv)
- [latent_publication_stats_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_stats_summary.csv)
- [latent_publication_family_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_family_summary.csv)
- [latent_publication_baseline_feasibility.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_baseline_feasibility.csv)
- [latent_publication_pc_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_pc_summary.csv)
- [latent_publication_control_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_control_summary.csv)
- [latent_publication_metadata.json](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_metadata.json)

This means the main work in Stage 2 should be:

- redesign plotting
- tighten explanatory text
- update README reproducibility guidance

### Final Main-Figure Concept

The refreshed main figure should be a static, publication-facing figure with **four panels**:

#### Panel A

**A. Full vine outperforms all usable baselines**

Use:

- Graphical Lasso
- Gaussian SSM
- Gaussian copula
- `1`-trunc vine

Use `Baseline NLL - Full-vine NLL`, so positive means full vine is better.

Visual plan:

- one dot-and-interval per baseline for the session-level mean paired difference
- light gray jittered session points behind each baseline
- no TVGL in the panel itself

TVGL handling:

- keep it out of the plot because it did not yield usable session-level latent-static output
- state this clearly in caption/overview text
- and point to the feasibility table

#### Panel B

**B. Gain includes a higher-order component**

Use:

- Pairwise non-Gaussian gain = `Gaussian -> 1-trunc`
- Higher-order gain = `1-trunc -> full` = `TC_higher`

Visual plan:

- two dot-and-interval columns
- light session points
- emphasized mean + bootstrap CI

#### Panel C

**C. Strongest signal in recruited/non-targeted space**

Compare:

- targeted
- mixed targeted + non-targeted
- non-targeted

Use a paired session display for:

- full-vs-Gaussian
- `TC_higher`

Preferred layout:

- two small aligned subpanels inside Panel C
- left: overall gain
- right: higher-order gain

That keeps the biological source-space question central without overloading one axis.

#### Panel D

**D. Dependence is heavy-tailed and asymmetric**

Replace dose in the main figure with grouped dependence classes:

- independence
- Gaussian-like elliptical
- heavy-tailed elliptical
- lower-tail asymmetric

Preferred content:

- main non-targeted latent result by default
- if visually clean, optionally add source-space split as a second grouped view
- but avoid making the panel too dense

Preferred visual:

- composition plot with a clean white background and restrained colors
- likely horizontal stacked bars or point-ranges by grouped class
- choose the cleaner of the two after inspecting readability

### Main Figure Versus Supplement

Main figure:

- Panel A baseline comparison
- Panel B pairwise-vs-higher-order decomposition
- Panel C source-space comparison
- Panel D grouped dependence type

Supplement:

- dose robustness moves out of the main figure
- catch/control remains supplement-only because it is not clean enough
- dynamic/history stays out entirely in this pass
- extra family breakdowns by dose stay in supplement
- extra PC interpretability diagnostics stay in supplement

Expected refreshed supplement figures:

- a dose robustness supplement figure built from current pooled/session dose summaries
- a family supplement figure that can include by-dose grouped families and raw stable-family detail
- the current dynamic supplement can stay on disk but does not need redesign in this pass

### Figure Style Changes

Stage 2 should update the figure style so it reads like a paper figure rather than a diagnostic plot.

Planned style changes:

- white background
- thinner gray session lines
- smaller gray session points
- only mean + CI emphasized in color
- concise scientific panel titles
- cleaner axis labeling with minimal rotation
- consistent positive-is-better y-axis semantics where applicable
- more compact legend handling
- slightly tighter panel spacing and margins

Preferred panel titles:

- **A. Full vine outperforms all usable baselines**
- **B. Gain includes a higher-order component**
- **C. Strongest signal in recruited/non-targeted space**
- **D. Dependence is heavy-tailed and asymmetric**

### Statistical Presentation To Reuse

No new inference framework is needed in this pass.
Reuse the current publication-ready summaries:

- session as the replication unit
- session-level paired points/lines
- bootstrap `95%` confidence intervals
- sign-flip / paired permutation-style summaries already exported in:
  - [latent_publication_stats_summary.csv](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_stats_summary.csv)

Stage 2 should surface these statistics more clearly in captions/overview text rather than adding new tests.

### New Explanatory Markdown File

Stage 2 should create:

- `DALGLEISH_LATENT_ANALYSIS_OVERVIEW.md`

This file should be written for a new reader and should be pedagogical rather than notebook-like.

Planned sections:

1. What is the dataset?
2. What scientific question are we asking?
3. How is the data preprocessed?
4. Why do we use PCA?
5. What analyses are performed?
6. What hypotheses were tested?
7. What did we find?
8. What can we claim, and what can we not claim?
9. What figures and outputs are produced?
10. How to rerun the analysis

Important tone rule:

- explain the latent-state formulation in plain language
- explicitly say this is a **population-state** claim
- and explicitly say what is not being claimed:
  - no neuron-level motif claim
  - no behavior claim
  - no dynamic/time-history main claim

### README Update Plan

Stage 2 should add or revise a dedicated section such as:

- `Dalgleish latent-state publication figures`

It should include:

- environment/prerequisites
- exact command for:
  - `scripts/run_dalgleish_latent_publication_analysis.py`
- exact command for any new figure-refresh script if one is added
- output figure paths
- output CSV paths
- a note that the overview markdown lives at:
  - `DALGLEISH_LATENT_ANALYSIS_OVERVIEW.md`

### Likely Stage 2 Implementation

The cleanest Stage 2 path is likely:

- keep the existing analysis outputs as source tables
- add one lightweight figure-refresh script, for example:
  - `scripts/refresh_dalgleish_latent_publication_figures.py`
- have that script read the validated publication CSVs and redraw:
  - the refreshed main figure
  - refreshed supplement figures for dose/family if needed

This avoids rerunning the full latent analysis just to change figure design.

### Expected Stage 2 Outputs

At minimum:

- updated main publication figure
- updated supplement figure(s) for dose/family if needed
- updated [README.md](/Users/alessandro/Documents/github/DVC/README.md)
- new `DALGLEISH_LATENT_ANALYSIS_OVERVIEW.md`

Optional but useful:

- one small JSON/CSV manifest mapping panels to source tables and interpretations

### What This Pass Should Improve

If Stage 2 goes as planned, the result should be:

- a cleaner main figure centered on the static latent claim
- clearer separation between headline claims and secondary results
- more paper-like visual styling
- and a concise document that lets a new collaborator understand the whole Dalgleish latent analysis quickly

## Publication Figure Refresh And Overview: Results

### Objective Of This Pass

This pass did **not** rerun the latent benchmark.
It refreshed the publication-facing figures and added a reader-facing overview document using the existing validated publication outputs.

### Code And Documents Added Or Updated

New figure-refresh script:

- [refresh_dalgleish_latent_publication_figures.py](/Users/alessandro/Documents/github/DVC/scripts/refresh_dalgleish_latent_publication_figures.py)

New reader-facing overview:

- [DALGLEISH_LATENT_ANALYSIS_OVERVIEW.md](/Users/alessandro/Documents/github/DVC/DALGLEISH_LATENT_ANALYSIS_OVERVIEW.md)

Updated documentation:

- [README.md](/Users/alessandro/Documents/github/DVC/README.md)

### Refreshed Figure Set

Main figure:

- [fig_latent_publication_final.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_final.png)

Supplement figures produced in this refresh:

- [fig_latent_publication_dose_supplement.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_dose_supplement.png)
- [fig_latent_publication_family_supplement.png](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/plots/fig_latent_publication_family_supplement.png)

Figure/source mapping:

- [latent_publication_figure_panel_map.json](/Users/alessandro/Documents/github/DVC/results/stimulation_exp_benchmark/data/latent_publication_figure_panel_map.json)

### Main Figure Layout Implemented

The refreshed main figure now follows the intended static publication story:

- **Panel A:** full vine versus all usable baselines
- **Panel B:** pairwise-flexible versus higher-order gain
- **Panel C:** source-space comparison with:
  - full-vs-Gaussian
  - `TC_higher`
- **Panel D:** grouped dependence type by source space

Main design choices:

- white background
- smaller gray session points
- thin gray session lines where appropriate
- mean plus bootstrap CI emphasized in color
- scientific message titles rather than diagnostic titles
- dose removed from the main figure
- dynamic/history not touched in this refresh

### What Stayed In Supplement

This refresh intentionally moved or kept the following out of the main figure:

- dose robustness
- catch/control
- dynamic/history analyses
- extra family detail beyond the grouped dependence-type panel

That makes the main figure cleaner and keeps the headline focused on the static latent-state result.

### Overview Markdown Added

The new overview document explains, in plain language:

- what the Dalgleish dataset is
- what scientific question the latent-state analysis asks
- how preprocessing and train-only PCA work
- why the analysis is done in latent population space
- what the main analyses are
- what was found
- what can and cannot be claimed
- which outputs and scripts matter

This is meant to be the fastest onboarding document for a new collaborator or reader.

### README Update

The README now includes:

- the publication analysis command
- the figure-refresh command
- where the refreshed figures are written
- where the overview markdown lives
- and where the panel-map manifest lives

### Scientific Content Status After The Refresh

This pass did not change the underlying scientific result.
It changed how that result is presented.

The main scientific story remains:

- full vine beats all usable baselines in the static latent-state setup
- the gain includes both:
  - pairwise-flexible / non-Gaussian gain
  - higher-order / conditional gain
- the strongest signal is in non-targeted or mixed latent space, not targeted-only
- the fitted dependence is best described as heavy-tailed plus lower-tail-asymmetric, not Gaussian-like

Dynamic/time-history remains secondary and was intentionally left out of this pass.
