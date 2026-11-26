# E80. Fourth-order episodes without constant triples

Written on 26 November 2026, before any dataset of this study was generated.

## Question

In the 50-variable challenge E31 (`e31_challenge.py`, frozen estimator `vinemodel2` of tag `vinemodel-a2`), every
fourth-order episode came from one law, `nested`, in which each episode sits on a pair that also carries a constant
third-order term with one of the episode's two modulators, beside four further constant triples. Only 4 of its 24
fourth-order episodes met the combined endpoint of Fig 3E (subset retained, change declared, every active window
recovered), against 104 of 138 third-order episodes in the other laws. Is the low count a property of fourth-order
episodes for this estimator at these sample sizes, or of the nested design?

## Design

* **Law `quartets`** (`e80_laws.py`): `nested` with every constant third-order term removed. The same background
  (five blocks of a hub and nine leaves, Gaussian chain of hubs that strengthens at window 3, rising block at
  window 6), the same four planted pairs, the same two modulators per pair, the same episode (windows 4 to 7 of
  12), the same strength g = 0.6 on the Fisher-z scale and the same roles and sizes (training, validation and
  confirmation n per window, 1,000 test observations per window). Nothing else is planted. For any seed the
  quartets are those that `nested` would plant with that seed; `e31_laws.py` is not changed.
* **Estimator and settings**: those of E31, unchanged. `e80_quartets.py` calls the function `run` of
  `e31_challenge.py` with only the data generator replaced; `vinemodel2.fit` with its defaults, 399 halves,
  `max_order = 4` (as for `nested`), 16 workers per dataset, the strict detector off. Jobs run from a pinned copy of
  the tagged commit `beffc6f` (`vinemodel-a2`) to which only the E80 files are added, with the Python environment of
  E31 (`Code/DVC/.venv`).
* **Primary datasets (fresh seeds)**: n = 300 and n = 1000; three planted datasets (seeds 991800, 991801, 991802)
  and one with nothing planted (991803, g = 0) per n: 8 datasets, 24 planted fourth-order episodes.
* **Secondary datasets (paired, a diagnostic)**: the law `quartets` on the three planted seeds of E31 (990000,
  990001, 990002) at both n: 6 datasets. With the same seed the sampler draws the same random numbers
  (`e26_laws.sample` draws all innovations before it couples the planted pairs), so each dataset equals E31's
  `nested` dataset of that seed except in the coupling of the planted pairs; each quartet episode is compared with
  its own outcome in E31. The datasets with nothing planted would be identical to E31's and are not run again.
* **Scoring** (as `v3/paper/worked_examples.py`, `challenge`, and `v3/paper/verify_challenge.py`): for every planted
  term, retained (`installed`), change declared (`change`), every active window (`installed` and
  `windows_on == windows_true`); the combined endpoint requires all three. Per dataset, terms that are not planted:
  retained and declared changing; held-out gain of the assembled over the simplified vine on the test windows.
  `e80_summary.py` reads the outputs, the 16 E31 `nested` receipts (checked against the hashes in
  `v3/paper/evidence/worked.json`) and writes `verification/e80_*.tsv` and `verification/e80_summary.json`.

## Expectations (primary datasets)

* (q1) No dataset fails.
* (q2) Nothing planted (2 datasets): no change declared; at most 2 retained terms in total.
* (q3) Every planted dataset (6): the assembled vine has a higher held-out log density than the simplified vine.
* (q4) Fourth-order episodes retained: at least 9 of 12 at n = 1000 (the count of `nested`); no prediction at n = 300.
* (q5) Fourth-order episodes declared changing: at least 6 of 12 at n = 1000; no prediction at n = 300.
* (q6) Retained terms that are not planted: a change is declared for at most 1 over the 6 planted datasets.

## Decision rule (primary datasets, combined endpoint over both sizes, 24 episodes; `nested`: 4 of 24)

* 12 or more of 24: the constant triples of the nested design were the main obstacle (the design).
* 8 or fewer of 24: fourth-order episodes remain hard for this estimator at these sizes without the triples
  (the order).
* 9 to 11: mixed; the conclusion is drawn endpoint by endpoint and size by size.

The paired datasets are reported beside the rule and do not change it.

## Results (26 November 2026, after the run)

**Run.** All 14 datasets completed with the frozen E31 estimator. Each generating seed planted the same four fourth-order episodes as the corresponding nested design. The paired simulations used the original seeds and differed only in the eight coupled leaves for the checked seed. The scoring script is `e80_summary.py`. The episode-level results distributed with the study are in `figures/evidence/e80_episodes.tsv`.

**Fourth-order episodes** (k of 12 per size):

| Law | n | Retained | Declared changing | Every active window | All three |
|---|---:|---:|---:|---:|---:|
| `quartets` (seeds 991800 to 991802) | 300 | 10 | 8 | 8 | 8 |
| | 1000 | 12 | 12 | 12 | 12 |
| `nested` (E31) | 300 | 6 | 0 | 4 | 0 |
| | 1000 | 9 | 6 | 6 | 4 |
| `quartets`, paired (E31's seeds 990000 to 990002) | 300 | 10 | 10 | 8 | 8 |
| | 1000 | 11 | 11 | 11 | 11 |

Across both sample sizes, 20 of 24 episodes met all three criteria in fresh simulations without the constant triples, compared with 4 of 24 in the original nested design. On the original seeds the count without the triples was 19 of 24. Recovery improved in every matched dataset. Across the two sample sizes the gains were five, four and six additional episodes for the three seeds. Because four episodes share each fitted population, the manuscript reports these counts descriptively.

**Paired, episode by episode** (the same data apart from the constant triples; counts both, with triples only,
without triples only, neither; exact McNemar test): all three endpoints 4, 0, 15, 5 (p = 0.0001); declared changing
6, 0, 15, 3 (p = 0.0001); every active window 9, 1, 10, 4 (p = 0.012); retained 12, 3, 9, 0 (p = 0.15).

**Datasets.** Planted, fresh seeds: the assembled vine beat the simplified vine on held-out data in 6 of 6 (gain
0.029 to 0.078 nats per observation; in `nested`, where the persistent triples carry most of the gain, 0.60 or more),
and the simplified vine was never returned; 16 unplanted terms were retained in 4 of 6 datasets, mostly fourth-order
terms on the edge of a planted episode or sharing two of its variables, and none was declared changing. Nothing
planted: no term at n = 300; at n = 1000 two terms within block 0 (a triple and a quartet), neither declared changing.

**Expectations.** (q1) to (q6) hold; (q2) at its bound (two retained terms over the two datasets with nothing planted).

**Decision.** The count of 20 crossed the prespecified threshold of 12. The persistent terms in this design accounted for much of the difficulty in E31.

**Where `nested` lost its episodes** (paired receipts). Of its 15 retained fourth-order episodes, 6 were declared
changing, all at n = 1000. In 8 of the 9 others the change test of the vine's own orientation gave p below 0.05 (most
at the floor set by 399 halves); the declaration failed because the joint fit with the constant triple on the same
pair (5, all at n = 300) or another orientation (3) gave a larger p, which then had to pass the false-discovery rule
over the 11 to 18 retained terms of the dataset (3 to 11 without the triples). Retention was also lower with the
triples (15 against 21 of 24). Without them every retained episode was declared changing.

**Caveats.** Twelve episodes per size; one strength (0.6), one episode (windows 4 to 7), one background; two datasets
with nothing planted; the strict detector was not run. The result concerns this estimator's decision rule (joint fits
with terms that share the pair, a false-discovery rule over retained terms), which a constant companion term makes
harder to pass; it does not show that fourth-order episodes are as easy as third-order ones in every design.
