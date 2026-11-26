"""Shared pieces of the experiments: penalty grid, validation selection, scan summaries."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvc_package.modulation import cvine, families as fam, scan  # noqa: E402

GRID = [dict(switch=s, drift=d, activation=a) for s, d, a in itertools.product((0.02, 0.08, 0.2), (0.0, 0.02), (0.0, 0.02))]
STATES = fam.default_states(('gaussian', 'student', 'clayton', 'gumbel', 'frank'), rotations=False)   # benchmark set
LEVEL = 0.05


def training_transform(train, other):
    """Map held-out raw values through each window's training empirical distribution (no held-out ranks)."""
    out = np.empty(other.shape)
    for t in range(train.shape[0]):
        for j in range(train.shape[2]):
            s = np.sort(train[t, :, j])
            out[t, :, j] = (np.searchsorted(s, other[t, :, j], side='right') + 0.5) / (s.size + 1.0)
    return np.clip(out, 1e-6, 1 - 1e-6)


def select(train, validation, max_orders=(2,), order=None, states=STATES, **kwargs):
    """Fit every penalty candidate at every maximum order once and choose by validation score.

    Returns {max_order: (validation score, model, options)}.  For max_order > 2 the simplified candidates compete
    too, so modulation is kept only when it improves the validation score."""
    fits = {}
    for order_now in sorted(set(max_orders) | {2}):
        for index, options in enumerate(GRID):
            model = cvine.fit(list(train), order=order, states=states, max_order=order_now, **options, **kwargs)
            fits[(order_now, index)] = (cvine.score(model, list(validation), eps=1e-6), model, options)
    best = {}
    for max_order in max_orders:
        for (order_now, index), item in sorted(fits.items()):
            if order_now in (2, max_order) and (max_order not in best or item[0] > best[max_order][0] + 1e-12):
                best[max_order] = item
    return best


def scan_summary(records, steps, level=LEVEL, null=None):
    """Familywise decisions from an anchored scan: per-window Bonferroni and interval scan over time."""
    z = np.array([r['z'] for r in records])                       # (R, T)
    threshold = scan.bonferroni_threshold(z.size, level)
    strength, start, stop = scan.interval_scan(z.T)
    null = scan.interval_scan_null(steps) if null is None else null
    interval_threshold = float(np.quantile(null, 1 - level / len(records)))
    return dict(threshold=threshold, interval_threshold=interval_threshold,
                records=[dict(subset=list(r['subset']), order=r['order'], shape=r['component'][0],
                              z=[round(float(v), 4) for v in r['z']],
                              windows=[int(t) for t in np.nonzero(np.abs(r['z']) > threshold)[0]],
                              interval=[float(strength[i]), int(start[i]), int(stop[i])],
                              interval_reject=bool(strength[i] > interval_threshold)) for i, r in enumerate(records)])
