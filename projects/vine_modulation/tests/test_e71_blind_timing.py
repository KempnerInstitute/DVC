"""The blind timing test of experiments/e71_blind_timing.py on a made-up recording whose replay changes half way."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e71_blind_timing", HERE.parent / "experiments" / "e71_blind_timing.py")
e71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e71)
spec = importlib.util.spec_from_file_location("toy", HERE / "test_e69_replay_vine.py")
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def test_a_replay_change_is_found_where_it_is_without_the_boundary():
    rng = np.random.default_rng(1)
    rec = toy.recording(rng, events=(2000, 2000), shares=(0.1, 0.45))
    report, score, block, rank = e71.analyse(rec)
    assert abs(report["profile_best"]["location"] - 0.5) <= 0.05 and report["profile_best"]["gain_over_one_law"] > 20
    assert all(abs(f["replay_location"] - 0.5) <= 0.10 for f in report["folds"])
    z = report["blind_minus_none"]["total"] / np.sqrt(report["blind_minus_none"]["variance"])
    assert z > 2
    assert report["given_minus_blind"]["total"] < 2 * np.sqrt(report["given_minus_blind"]["variance"])
    assert set(score) == {"none", "blind", "given", "persistent", "shared", "class"} and all(np.isfinite(v).all() for v in score.values())
    assert all(m in e71.LOCATIONS for m in (f["modulation_location"] for f in report["folds"]))
