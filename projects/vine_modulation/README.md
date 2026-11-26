# Dynamic vine copulas for changing interactions

This DVC project reproduces the examples, figures and analyses in *Tracking changes in higher-order dependence with dynamic vine copulas*. The study by Houman Safaai and Alessandro Marin Vargas runs on a CPU. Begin with the examples to see the difference between a changing pair and a changing modulation of that pair. Each module below implements a distinct statistical quantity used in the article.

## Quick start

From the root of the DVC checkout or the extracted S1 Code archive, use Python 3.10:

~~~sh
python -m venv .venv-study
. .venv-study/bin/activate
python -m pip install -r projects/vine_modulation/requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
cd projects/vine_modulation
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python run_examples.py continuous continuous.json
python run_examples.py spikes spikes.json
python -m pytest -q tests
~~~

The estimator is part of the existing `dynamic-vine-copulas` distribution under `dvc_package.modulation`. The command above uses its source directly with the numerical versions used for this paper. A normal installation of DVC also exposes these entry points. Its wheel was checked under NumPy 1.26.4, and both worked examples reproduced their reference results at that version. The study scripts live in this project folder. S1 Code preserves these repository paths and includes the DVC licence.

The first example returns true and fitted coefficient paths for a Gaussian conditional pair in 12 windows. In the reversal example, the conditional association changes sign while every pairwise distribution stays fixed. The spike example constructs two three-cell tables with exactly the same two-cell counts and returns their known log-odds change and the conditional estimate. A 1 in a pattern denotes participation. Outcomes in both examples are simulated.

## Input and output

- **Continuous population.** Use `dvc_package.modulation.fit(train, validation, confirm, test, ...)` with sequences of observations-by-variables arrays, one per window and split. Use the same column order throughout and independent splits or contiguous separated time blocks for serial data. The default `max_order=3` searches single modulators. Set `max_order=4` to include products of two modulators; the study estimates their localization accuracy separately. The fit selects a C-vine order, fits its background, screens conditioning features and compares persistent and changing coefficients. A component refers to an edge, its conditioning feature and this order. The fitted report keeps the selected components, half-sample change tests and held-out density scores. `experiments/e31_challenge.py` shows the complete call.
- **Binary place-cell trios.** `dvc_package.modulation.pooled_change(before, after, sets)` takes two event-by-cell binary matrices and a matrix of pair or trio indices. It returns a shared log-odds shift, its standard error and its information. `prepare`, `combine` and `contrast` compare classes that can share cells and pool recordings. They preserve the covariance of the estimated class shifts. Optional activity strata condition on the number of other active cells. This is the marginal fixed-pair-margin contrast in Fig 4.
- **Whole ripples.** `experiments/e69_replay_vine.py` scores each complete binary pattern as a sequence of conditional Bernoulli predictions in fixed cell order. Its third-cell coefficients share information across trios. The unit of prediction is the entire ripple, and its log score has different units from a marginal trio's log odds. `experiments/e71_blind_timing.py` searches for a boundary on training blocks. `experiments/fold_scale_audit.py` compares original and training-fold standardization, recording by recording. The schedule control scores a fixed one-third split on the same held-out events. The fold-calibration summary compares paired simulation replicates under the original and training-fold scaling conventions. The optional `common` argument to the whole-ripple driver measures a shared change per additional active pair in both classes. `experiments/check_shared_scale.py` audits the units of the plant and `experiments/summarize_common_count.py` scores the matched calibration and recording predictions.

## Reproduce the seven figures

Run these commands from `projects/vine_modulation/` after the quick start. The figures use the included numerical inputs. Original recordings are needed only to repeat data extraction or model fitting.

~~~sh
mkdir -p plots
python figures/draw_fig1_concept.py plots/concept.pdf plots/fig1_source.csv
python figures/draw_fig2_worked.py plots/worked_examples.pdf plots/fig2_source.csv
python figures/draw_fig3_recovery.py plots/episode_recovery.pdf plots/fig3_source.csv
python figures/draw_fig4_hippocampus.py plots/hippocampus_terms.pdf plots/fig4_source.csv
python figures/draw_fig5_checks.py plots/hippocampus_checks.pdf plots/fig5_source.csv
python figures/draw_fig6_replay.py plots/hippocampus_replay.pdf plots/fig6_source.csv
python figures/draw_fig7_whole.py plots/whole_ripples.pdf plots/fig7_source.csv
python figures/figure_source_data.py plots/figure_source_data.xlsx
python figures/make_hippocampus_table.py plots/animal_contrasts.tex
python figures/make_calibration_table.py plots/change_calibration.tex
~~~

The workbook includes the plotted estimates, the simulated example trajectories, the timing profiles and the empirical pairwise-model ranges. The small input file `figures/evidence/figures/hippocampus_examples.json` also contains the illustrated place fields, the running rate maps of every place cell on one running direction, spike rasters and decoded location profiles. All figure scripts share `figures/figstyle.py`: text in the Helvetica core font at 8 to 12 pt, formulas drawn as vector outlines, and a blue and orange palette in which marker shapes, line styles or labels, never colour alone, identify each series. Selection rules and hashes are in the accompanying `receipt.json`. With the study outputs available, run `python figures/export_figure_data.py RESULTS_ROOT OUTPUT_DIRECTORY` to regenerate these inputs.

| Figure | From observations to the plotted quantities |
| --- | --- |
| 1–2 | `worked_examples.py` generates the simulated continuous examples. `run_examples.py` illustrates the exact binary contrast with fixed pair margins. |
| 3 | The E17 and E17b drivers fit identical simulated datasets. `figures/known_edge_timing.py` scores each saved path. The optional GAM comparator uses R with gamCopula 0.0-8, VineCopula 2.6.1, mgcv and jsonlite. Pass the R library directory as the `RLIBRARY` argument to `experiments/e17b_addendum.py`. `experiments/e31_challenge.py` fits one 50-variable population. All 64 datasets are identified in `figures/evidence/worked.json`. |
| 4 | `real_data/hippocampal_census.py`, `hippocampal_place_fields.py` and `hippocampal_study.py` extract and analyze the first cohort. `real_data/dryad_cohort.py` handles the second. |
| 5 | The E64 checks stratify ripple events by state or stillness. `experiments/e65_pairwise_fit.py` fits the session-level pairwise models and simulates the reference distributions. |
| 6 | The two `*_posthoc_content.py` drivers fit and simulate the track-stretch replay model. The exported recording examples identify the source session, trios and ripples. |
| 7 | `experiments/e69_replay_vine.py` scores whole held-out ripples. `experiments/e71_blind_timing.py` fits the change-point profiles. The fold and schedule audits described above supply the held-out comparisons. |

## Repeat one population fit

Run this example from the study folder after the quick-start setup. It simulates a 50-variable population, searches for the interacting variables, selects trajectories and scores the complete distribution on the held-out data. Use fewer workers on a small machine without changing the seed or the number of half-sample fits.

~~~sh
python experiments/e31_challenge.py population.json step 300 0.6 990001 4 399 0
~~~

The original receipt has six planted terms. The average held-out log density in this dataset is 7.0560363 nats for the assembled vine and 6.8880916 for its simplified background. A candidate may be screened, retained or declared changing. These are separate outcomes. The full figure counts an episode only when its subset, change decision and active windows all agree with the plant.

Each public source export contains a manifest with the Git revision and a SHA-256 checksum for every file. The input receipts retain the checksums recorded when the numerical results were generated. From a full DVC checkout, run `python projects/vine_modulation/export_code.py . HEAD methods_code.zip` to export the files listed in `release_files.txt`. The README is also placed at the archive root. The study design, calibration exceptions and limits of the estimators are summarized in `projects/vine_modulation/PROTOCOL.md`.

## From the public recordings

The original recordings are available through [DANDI, version 0.261106.0236](https://doi.org/10.48324/dandi.000044/0.261106.0236) and [Dryad, version 13](https://doi.org/10.5061/dryad.ksn02v76h). Preserve the distributed file and session names. DANDI contains NWB session files. Dryad supplies the source MATLAB and session files referenced in `real_data/dryad_cohort.py`. The supporting methods document eligibility and the corrected animal identities. The paper uses the two datasets' original sleep or rest definitions.

For one DANDI session, substitute its input and choose separate output directories. The extraction steps expect new output directories.

~~~sh
python real_data/hippocampal_census.py SESSION.nwb CENSUS
python real_data/hippocampal_place_fields.py SESSION.nwb FIELDS
python real_data/hippocampal_study.py session SESSION.nwb CENSUS FIELDS STUDY
~~~

For a Dryad session:

~~~sh
python real_data/dryad_cohort.py census DATA_FOLDER SESSION CENSUS
python real_data/dryad_cohort.py fields DATA_FOLDER SESSION FIELDS
python real_data/dryad_cohort.py study DATA_FOLDER SESSION CENSUS FIELDS STUDY
~~~

The `combine` and `claims` commands in the respective study drivers pool the resulting recording-level products. The `fit` command of each replay driver accepts the source data folder, census root, fields root, output folder and session identifier. Its `study`, `simulate` and `summary` commands produce the recorded, simulated and adjusted class contrasts. For population prediction use `python experiments/e69_replay_vine.py study CONTENT_FOLDER OUT_FOLDER NAME [CENSUS_ROOT]` followed by `summary OUT_FOLDER`. The header of each driver gives the command for each stage and describes its decisions. Computation and storage for raw recordings exceed those of the worked examples.

`MANIFEST.json` gives the exact Git revision, source path and SHA-256 of every distributed file. The archive is generated from the DVC library commit. The project depends on the implementation in `src/dvc_package/modulation/` and contains the generators, recording protocols, plots and their small numerical inputs. S1 Text records the data splits, nulls, calibration exceptions and limits relevant to interpreting the output.
