# E79. Is the persistent whole-ripple three-cell gain population synchrony?

Written on 26 November 2026, before anything of E79 was run on the recordings or on simulations.
Driver: `experiments/e79_synchrony.py` (imports `e69_replay_vine.py` unchanged).

## Question

In E69/E73 each place cell's logit, given the cells before it in field order, is the replay law's prediction plus
pair edges (persistent J and their changes a, b), then a persistent three-cell modulation `d_O N_O + d_S N_S`, a
shared change `c` and a class-specific change `Delta`. `N_O(i)` and `N_S(i)` count the active earlier pairs that
would close an overlapping or a separate trio with cell `i`, in training-fold units (section 73). In the independent
cohort the persistent modulation gained +39.82 nats (blocked SE 11.76) over 17,436 held-out ripples, positive in
8 of 9 sessions; in Achilles +2.75 (3.00).

Both counts grow roughly with the square of the number of active earlier cells. If ripples differ in overall
participation more than the replay law (five strengths) and the pair edges allow, any count of active earlier pairs
predicts the next cell, whatever the class. The persistent gain could then be class-independent population
synchrony rather than a three-cell term of the classes.

## Design

Same 22 recording units, ripples, field order, classes, ten consecutive blocks per rest, five outer folds (block m in
fold m mod 5), replay laws refitted to the training folds of each rest, pair penalty TAU chosen exactly as in E69
(`pairs` on three training folds, scored on the fourth), training-fold scales for every count (section 73), weak
penalty (SD 10) on every added coefficient, held-out scores of whole ripples, blocks as units within each rest.

New covariates of cell `i`, from the cells before it in field order (`k(i)` = number of active earlier cells):

- `N_all(i) = k(i)(k(i) - 1)/2`, all active earlier pairs whatever their class (primary synchrony covariate),
  divided by its training-fold standard deviation like `N_O` and `N_S`;
- `k, k^2, k^3`, each divided by its training-fold standard deviation (sensitivity: a flexible function of the
  number of active earlier cells).

Nested variants, each fitted by penalised maximum likelihood to the four training folds:

| Variant | Adds | Comment |
|---|---|---|
| replay | cell intercepts on the replay logit | as E69 |
| pairs | J, a, b | as E69 (identical fit, identical TAU) |
| persistent0 | d_O N_O + d_S N_S on `pairs` | E69's persistent model, recomputed as a reproduction check |
| sync | e N_all on `pairs` | synchrony |
| common | d (N_O + N_S) on `sync` | one trio coefficient for both classes |
| persistent | d_O N_O + d_S N_S on `sync` | the classes' persistent modulation beyond synchrony |
| shared | c u (N_O + N_S) on `persistent` | shared change, as E69 |
| class | Delta u N_O on `shared` | class-specific change, as E69 |
| sync_k | g1 k + g2 k^2 + g3 k^3 on `pairs` | sensitivity |
| persistent_k | d_O N_O + d_S N_S on `sync_k` | sensitivity |

Held-out comparisons (first minus second variant; total over ripples, blocked SE, sessions positive):
`synchrony` = sync - pairs (i); `modulation` = persistent - sync (ii); `common_modulation` = common - sync;
`class_difference` = persistent - common (do the classes need separate persistent coefficients?);
`time_varying` = shared - persistent and `class_change` = class - shared (iv); `modulation_e69` = persistent0 -
pairs (reproduction); `synchrony_k` = sync_k - pairs and `modulation_k` = persistent_k - sync_k (sensitivity).

Coefficients (iii): `persistent`, `persistent0` and `persistent_k` are also fitted to all ripples of each recording
(replay laws on all ripples of each rest, TAU the most frequent of the five folds' choices, scales from all
ripples). Standard errors of `d_O`, `d_S`, `d_O - d_S` (and `e`) are blocked sandwich errors: the penalised Hessian
with the cell intercepts and pair edges profiled out (Schur complement), and the efficient scores summed within each
of the 20 blocks, centred within each rest, with the factor B/(B-1). Recordings are pooled within a group by
inverse-variance weighting.

Checks on simulated cohorts of the nine second-cohort sessions (the generator of E69's `check`: a stretch and a
strength from each ripple's posterior under the stored law of its rest, then each cell in field order), new seed
990790, 8 cohorts per scenario, scored with `replay, pairs, persistent0, sync, common, persistent`:

- `replay`: nothing planted;
- `synchrony`: a class-independent term `0.15 N_all` (per recorded SD of `N_all`), no class modulation;
- `persistent_class`: a persistent `0.15 N_O` (per recorded SD of `N_O`), no synchrony, no change.

## Expectations and decision rule (written before the runs)

Study:

- (s0) Reproduction: `pairs - replay` and `persistent0 - pairs` equal E73's totals (`e69/study_fold_20261125`)
  within 1e-6 nats per recording.
- (s1) `synchrony` is reported in both groups. I expect it to be positive in the independent cohort.
- (s2) Decision for the independent cohort. The persistent gain **is synchrony** if `modulation` (persistent -
  sync) has blocked z < 1.96 and is below half of `modulation_e69`. It is **a three-cell term beyond synchrony**
  if `modulation` has z >= 1.96. It is **class-specific** only if, in addition, `class_difference` has z >= 1.96
  or the pooled `d_O - d_S` is positive with z >= 1.96; otherwise it is a term shared by both classes. A result
  between these (z < 1.96 but at least half of the E69 gain) is reported as unresolved.
- (s3) With synchrony in every model, `time_varying` and `class_change` keep the E73 conclusions (no positive
  comparison with z >= 1.96) in the independent cohort and in Achilles.
- (s4) The same decision is applied to `modulation_k` (k, k^2, k^3) as a sensitivity; a disagreement with (s2) is
  reported, and the primary covariate decides.
- Achilles is reported in the same terms; with E73's +2.75 (3.00) I expect nothing to decide there.

Checks (8 cohorts each; "crosses" means blocked z >= 1.96 for the cohort):

- (c1) `replay`: `synchrony` and `modulation` each cross in at most 1 of 8 cohorts.
- (c2) `synchrony` plant: `synchrony` crosses in at least 4 of 8; `modulation` (beyond synchrony) crosses in at
  most 1 of 8. Reported, not required: how often `modulation_e69` crosses (whether pure synchrony can pass for the
  classes' modulation without the control).
- (c3) `persistent_class` plant: `modulation` (beyond synchrony) crosses in at least 4 of 8, so the control does
  not remove a planted class term; `class_difference` reported.

If (c2) or (c3) fails, the study's decision (s2) is reported with that limitation.

## Results (added on 26 November 2026, after the runs)

Runs: study Slurm 63796930 (all 22 recordings, `RESULTS_ROOT/e79/study_v1`), check Slurm 63797882
(`RESULTS_ROOT/e79/check_v1`), conda Python 3.10 with NumPy 2.2.6 and SciPy 1.15.3. Tables:
`verification/e79_study_summary.json`, `e79_study_recordings.tsv`, `e79_study_coefficients.tsv`,
`e79_check_summary.json`; source checksums in `verification/e79_sources.sha256`.

(s0) holds: the replay, pair and E69 persistent scores equal E73's for every ripple (largest difference 0.0).

Independent cohort, nine sessions, 17,436 held-out ripples (total nats, blocked SE, sessions positive):

| Comparison | Gain | SE | z | Positive |
|---|---:|---:|---:|---:|
| synchrony (sync - pairs) | +10.67 | 4.67 | 2.28 | 8/9 |
| E69 persistent modulation (persistent0 - pairs) | +39.82 | 11.76 | 3.39 | 8/9 |
| modulation beyond synchrony (persistent - sync) | +33.49 | 10.52 | 3.18 | 6/9 |
| one coefficient for both classes (common - sync) | +8.29 | 6.79 | 1.22 | 5/9 |
| separate class coefficients (persistent - common) | +25.20 | 8.29 | 3.04 | 6/9 |
| shared change (shared - persistent) | +3.57 | 6.85 | 0.52 | 3/9 |
| class-specific change (class - shared) | -2.67 | 5.12 | -0.52 | 2/9 |
| synchrony as k, k^2, k^3 (sync_k - pairs) | +9.81 | 9.25 | 1.06 | 6/9 |
| modulation beyond it (persistent_k - sync_k) | +33.55 | 10.30 | 3.26 | 6/9 |

Pooled coefficients of `persistent`, per SD of each count, blocked sandwich errors: d_O -0.002 (0.004), d_S -0.057
(0.006; negative in all nine sessions), d_O - d_S +0.066 (0.008, z 8.2; positive in 8/9), e +0.036 (0.006). E69's
model without synchrony gave d_O +0.001 (0.004), d_S -0.043 (0.005), d_O - d_S +0.047 (0.007). Per-fold estimates
agree in sign in every fold of the larger sessions.

Achilles, 17,002 ripples: synchrony +0.33 (0.75), E69 persistent +2.75 (3.00), beyond synchrony +1.01 (3.32),
separate class coefficients -1.77 (1.21), shared change +4.21 (4.00), class-specific change -2.12 (2.32); d_O +0.007
(0.004), d_S -0.000 (0.007).

Decision (s2): in the independent cohort the persistent gain is a class-specific term beyond synchrony (z 3.18 beyond
synchrony; separate class coefficients needed, z 3.04; pooled d_O - d_S z 8.2). The class-specific part is a
negative modulation of the separate class: an active earlier pair whose fields lie apart from the predicted cell's
field lowers that cell's firing beyond replay, pairs and synchrony; the overlapping coefficient is near zero. (s4)
agrees. (s3) holds in both groups: no shared-change or class-specific-change gain. In Achilles there is no
persistent gain to explain (the summary's automatic label "synchrony" there only reflects a gain that was already
indistinguishable from zero).

Checks, 8 cohorts per scenario (crossings: blocked z >= 1.96): `replay`: synchrony 0/8, modulation 0/8 (c1 holds).
`synchrony` plant: synchrony 8/8; modulation beyond synchrony 0/8 (c2 holds); without the control, E69's persistent
modulation crossed 8/8 (mean +115 nats), so pure synchrony can pass for the classes' modulation when it is not in the
model. `persistent_class` plant: modulation beyond synchrony 8/8 (mean +631 nats) and separate class coefficients
8/8 (c3 holds).

Limits: one plant size (0.15 per SD) and 8 cohorts per scenario; coefficient errors use 20 blocks per recording and
fixed-effect pooling across recordings; synchrony is measured from the cells before each cell in field order, as the
sequential law requires.
