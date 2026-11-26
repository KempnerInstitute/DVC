"""E78: the content-only law with cell factors shared by the two rests, on small made-up sessions."""
import importlib.util
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


shared = module(PROJECT / "real_data" / "shared_factor_content.py", "shared_factor_content")
check = module(PROJECT / "experiments" / "e66_content_law_check.py", "e66_content_law_check")
content = shared.content


def session(seed, scale_after, events=400, cells=12):
    """A made-up session: the same fields in both rests, more replay after, factors multiplied by `scale_after`."""
    rng = np.random.default_rng(seed)
    truth = check.session_truth(rng, cells)
    before = check.true_law(truth, 0.25, check.MIDDLE)
    after_truth = dict(truth, factor=truth["factor"] * scale_after)
    after = check.true_law(after_truth, 0.5, check.MIDDLE)
    x = np.concatenate([check.sample(before, events, rng), check.sample(after, events, rng)])
    epoch = np.repeat(["PRE", "POST"], [events, events])
    d = dict(x=x, epoch=epoch, cells=np.arange(cells), rates=truth["rates"], kinds=truth["kinds"], window=0.1)
    return d, truth


def fitted(d):
    separate = {rest: content.Content(d["x"][d["epoch"] == rest][:, d["cells"]], d["rates"], d["window"]) for rest in content.RESTS}
    return separate, shared.fit_shared(d)


def test_shared_factors_nested_and_recovered():
    d, truth = session(7, 1.0)
    separate, joint = fitted(d)
    assert np.allclose(joint["PRE"].factor, joint["POST"].factor)
    row = shared.compare(d, separate, joint)
    assert row["gap"] > -1.0                          # the shared law is nested in the separate laws (up to EM convergence)
    assert row["aic_difference"] < 0                  # with equal true factors one factor per cell is enough
    assert np.corrcoef(np.log(joint["PRE"].factor), np.log(truth["factor"]))[0, 1] > 0.6


def test_changed_excitability_is_seen():
    d, _ = session(8, 3.0)
    separate, joint = fitted(d)
    row = shared.compare(d, separate, joint)
    assert row["aic_difference"] > 0                  # factors three times larger after: separate factors fit better
    assert row["factor_move"]["mean"] > 0.5
