"""E80.  Fourth-order episodes without constant triples, run through the frozen pipeline of E31.

Design, expectations and decision rule (written on 26 November 2026, before any dataset was generated; the same text
is in `E80_QUARTETS_ALONE.md`):
  Law `quartets` (`e80_laws.py`): the law `nested` of E31 with every constant third-order term removed (the same
  background, pairs, modulators, episode of windows 4 to 7 and strength g = 0.6; nothing else planted).
  Estimator: `e31_challenge.run` itself with only the data generator replaced (`vinemodel2.fit` defaults, 399 halves,
  max_order 4, strict detector off), from a pinned copy of the tag `vinemodel-a2` plus the E80 files.
  Primary datasets: n = 300 and 1000; seeds 991800, 991801, 991802 planted and 991803 with nothing planted.
  Secondary, paired: seeds 990000 to 990002 of E31 (planted only), each equal to E31's `nested` dataset of the same
  seed except in the coupling of the planted pairs.
  Expectations (primary): (q1) no dataset fails; (q2) nothing planted: no change declared and at most 2 retained
  terms in total; (q3) the assembled vine beats the simplified vine on held-out data in all 6 planted datasets;
  (q4) fourth-order episodes retained: at least 9 of 12 at n = 1000; (q5) declared changing: at least 6 of 12 at
  n = 1000 (no prediction at n = 300); (q6) at most 1 change declared for retained terms that are not planted.
  Decision (combined endpoint of Fig 3E over both sizes, 24 episodes; `nested`: 4 of 24): 12 or more, the nested
  design was the main obstacle; 8 or fewer, the order; 9 to 11, mixed.

Usage: e80_quartets.py OUT.json LAW N G SEED WORKERS [REPLICATES=399] [STRICT=0]     (LAW: quartets)
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import e31_challenge as frozen
import e30b_second_version as e30b
import e80_laws


def run(law, n, g, seed, workers, replicates=399, with_strict=False):
    """`e31_challenge.run` with the generator of `e80_laws`; everything else is the frozen code."""
    original = frozen.e31
    frozen.e31 = SimpleNamespace(generate=e80_laws.generate)
    try:
        return frozen.run(law, n, g, seed, workers, replicates, with_strict)
    finally:
        frozen.e31 = original


if __name__ == '__main__':
    target, law, n, g, seed, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    replicates = int(sys.argv[7]) if len(sys.argv) > 7 else 399
    with_strict = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
    summary, edges = run(law, n, g, seed, workers, replicates, with_strict)
    target.write_text(json.dumps(dict(summary=summary, edges=edges), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    e30b.describe(summary)
