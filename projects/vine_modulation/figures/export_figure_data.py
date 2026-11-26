"""Export the small data that the redrawn figures read from the study outputs, with the hashes of every input.

    export_figure_data.py RESULTS_ROOT OUT_FOLDER

RESULTS_ROOT contains the outputs of the study drivers.  Writes to
OUT_FOLDER (evidence/figures):

  known_edge_examples.tsv   the true baseline and modulation of two simulated edges of the known-edge study (E17/E17b)
                            and the slope path each of the four compared estimators fitted to them.  The examples are
                            fixed by rule, not by outcome: the first dataset (seed 980000) of the abrupt and of the
                            recurring design at strength 0.6 and 100 training observations per window.
  hippocampus_examples.json real recordings for the schematic panels of the hippocampal figures, from the second-cohort
                            session with the most place cells (rat 3, second session): the 4-cm rate maps of one
                            overlapping trio (the one whose smallest pairwise map correlation is largest) and of one
                            separate trio (the first, in the stored order, whose three fields peak on one running
                            direction); and three ripples of the rest after the tracks, chosen as the three with the most
                            active place cells among those whose most probable state under the fitted replay law is a
                            stretch of about 40 cm or 1 m with posterior probability at least 0.8, each with its place
                            cells' spikes in ten 10-ms steps, the posterior over positions and the running maps of its
                            active cells on the decoded direction; and the share of ripples after the tracks by the kind of
                            their most probable state.
  timing_profiles.tsv       for every recording, the fit of two replay laws split at each candidate change point minus
                            the fit of one law (E71, in sample), with the rank of the experience.
  e64_..., e65_..., e66_... copies of the project's verification tables for the hippocampal checks (sleep state and
                            stillness matching, pairwise models, simulated checks of the replay-relative estimate).
  receipt.json              SHA-256 of every input and output, the selection rules and the git revision of the code.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent
# Resolve the project in either supported distribution layout.
PROJECT = next(p for p in (PAPER.parents[1] / "projects" / "vine_modulation_20261109", PAPER.parent) if (p / "real_data").is_dir())
MODELS = ("joint", "switching", "smooth_x", "gamml")          # the four estimators of the paper's comparison
EXAMPLES = (("abrupt", 0.6, 100, 980000), ("recurring", 0.6, 100, 980000))
SESSION = "RAT3_SESS2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(loaded)
    return loaded


def known_edge(root, out, inputs):
    rows = ["design\tstrength\tn\tseed\twindow\ttrue_baseline\ttrue_modulation\t" + "\t".join(MODELS)]
    for part in ("n100",):
        original = root / f"e17_joint_trajectories_{part}.json"
        addendum = root / f"e17b_addendum_{part}.json"
        inputs += [original, addendum]
        first = {(r["design"], r["g"], r["n"], r["seed"]): r for r in json.loads(original.read_text())}
        second = {(r["design"], r["g"], r["n"], r["seed"]): r for r in json.loads(addendum.read_text())}
        for key in EXAMPLES:
            a, b = first[key], second[key]
            arms = {"joint": a["arms"]["joint"]["slope"], "switching": a["arms"]["switching"]["slope"],
                    "smooth_x": b["arms"]["smooth_x"]["slope"], "gamml": b["arms"]["gamml"]["slope"]}
            for w in range(len(a["c"])):
                rows.append("\t".join(map(str, [key[0], key[1], key[2], key[3], w + 1, f"{a['base'][w]:.6f}", f"{a['c'][w]:.6f}",
                                                *[f"{arms[m][w]:.6f}" for m in MODELS]])))
    (out / "known_edge_examples.tsv").write_text("\n".join(rows) + "\n")


def map_blocks(kinds):
    """For each state column: (map, bin, kind) with kind in 0..4; bin is -1 for a whole direction or none."""
    blocks, m, k_prev, i = [], 0, None, 0
    for kind in kinds:
        if kind in (0, 1, 2):
            if kind != k_prev:
                i = 0
            blocks.append((m, i, int(kind)))
            i += 1
        elif kind == 3:
            blocks.append((m, -1, 3))
            m += 1
        else:
            blocks.append((-1, -1, 4))
        k_prev = kind
    return blocks


def hippocampus(root, out, inputs):
    content = module(PROJECT / "real_data" / "dryad_posthoc_content.py", "dryad_posthoc_content")
    sequences = module(PROJECT / "real_data" / "replay_sequences.py", "replay_sequences")
    stored = root / "hippocampus" / "dryad_content_v1" / "sessions" / f"{SESSION}.npz"
    steps_file = root / "e72" / "extract_v1" / "steps" / f"{SESSION}.npz"
    inputs += [stored, steps_file]
    d = np.load(stored, allow_pickle=True)
    rates, kinds, cells = d["rates"], d["kinds"], d["cells"]
    x = d["x"][:, cells].astype(np.int8)
    y = np.load(steps_file)["y"]
    assert y.shape[:2] == x.shape and np.array_equal(y.max(2) > 0, x > 0)
    maps = sequences.maps_of(rates, kinds)                                   # list of (cells, bins) per running direction
    peak_map = np.array([int(np.argmax([m[c].max() for m in maps])) for c in range(len(cells))])
    peak_bin = np.array([int(np.argmax(maps[peak_map[c]][c])) for c in range(len(cells))])
    joined = np.concatenate(maps, axis=1)

    def corr(a, b):
        return float(np.corrcoef(joined[a], joined[b])[0, 1])

    position = {int(u): i for i, u in enumerate(cells)}                    # trio sets index all kept units; maps index place cells
    overlapping = [[position[int(u)] for u in t] for t in d["sets_trios_overlapping"]]
    smallest = [min(corr(t[0], t[1]), corr(t[0], t[2]), corr(t[1], t[2])) for t in overlapping]
    over = [int(c) for c in overlapping[int(np.argmax(smallest))]]
    separate = None
    for t in d["sampled_trios_separate"]:
        t = [position[int(u)] for u in t]
        if len({int(peak_map[c]) for c in t}) == 1:
            separate = [int(c) for c in t]
            break
    assert separate is not None
    patterns = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

    def counts(t, rest):
        rows = x[d["epoch"] == rest][:, t]
        return [int(np.all(rows == np.array(q), axis=1).sum()) for q in patterns]

    trio = lambda t: dict(cells=t, map=int(peak_map[t[0]]), correlations=[corr(t[0], t[1]), corr(t[0], t[2]), corr(t[1], t[2])],
                          rates=[maps[int(peak_map[t[0]])][c].tolist() for c in t], patterns=["".join(map(str, q)) for q in patterns],
                          counts_before=counts(t, "PRE"), counts_after=counts(t, "POST"))
    blocks = map_blocks(kinds)
    after = np.flatnonzero(d["epoch"] == "POST")
    law = content.Content.restored(rates, float(d["window"]), d["factor_POST"], d["weight_POST"])
    post = law.posterior(law.likelihood(x[after])).sum(0)                    # (events, states), summed over strengths
    top = post.argmax(1)
    top_kind = np.array([blocks[j][2] for j in top])
    best_state = post[np.arange(len(top)), top]
    eligible = np.flatnonzero(np.isin(top_kind, (1, 2)) & (best_state >= 0.8))
    chosen = eligible[np.argsort(-x[after][eligible].sum(1), kind="stable")[:3]]
    decoded_kinds = {name: float(np.mean(top_kind == k)) for k, name in enumerate(("bin", "within_20cm", "within_48cm", "whole_direction", "none"))}
    ripples = []
    order = np.lexsort((peak_bin, peak_map))                                 # place cells by running direction, then field peak
    for e in chosen:
        p = post[e]
        by_map = {}
        for j, (m, i, k) in enumerate(blocks):
            if k in (0, 1, 2):
                by_map.setdefault(m, np.zeros(maps[m].shape[1]))[i] += p[j]
        whole = {m: float(p[j]) for j, (m, i, k) in enumerate(blocks) if k == 3}
        m_top, centre, kind = blocks[top[e]]
        best = m_top
        fired_cells = np.flatnonzero(x[after[e]])
        overall = joined.max(1)
        fired_maps = [(maps[best][c] / max(overall[c], 1e-9)).tolist() for c in fired_cells]
        ripples.append(dict(index=int(after[e]), active_cells=int(x[after[e]].sum()), best_state_probability=float(best_state[e]),
                            none_probability=float(p[[j for j, b in enumerate(blocks) if b[2] == 4]].sum()),
                            position_posterior={str(m): v.tolist() for m, v in by_map.items()}, whole_direction=whole,
                            best_map=int(best), best_centre_bin=int(centre), best_half_width_bins=int((0, 5, 12)[kind]), best_kind=int(kind),
                            fired_maps_on_best_map=fired_maps,
                            steps=y[after[e]][order].tolist()))
    report = dict(session=SESSION, rest="after the tracks", place_cells=int(len(cells)), bin_cm=4, maps=[m.shape[1] for m in maps],
                  order=order.tolist(), peak_map=peak_map[order].tolist(), peak_bin=peak_bin[order].tolist(),
                  overlapping=trio(over), separate=trio(separate), ripples=ripples, decoded_kinds_after=decoded_kinds,
                  rules=dict(session="the second-cohort session with the most place cells",
                             overlapping="the overlapping trio whose smallest pairwise map correlation is largest",
                             separate="the first separate trio in the stored sample whose three fields peak on one running direction",
                             ripples="ripples after the tracks whose most probable state is a stretch of about 40 cm or 1 m with posterior probability at least 0.8; the three with the most active place cells"))
    (out / "hippocampus_examples.json").write_text(json.dumps(report) + "\n")


def timing(root, out, inputs):
    rows = ["recording\tboundary\tlocation\tsplit_minus_one_law"]
    files = sorted(f for f in (root / "e71" / "run_v1").glob("*.json") if f.stem != "summary")
    if len(files) != 22:
        raise ValueError(f"Expected 22 recording profiles, found {len(files)}")
    for f in files:
        inputs.append(f)
        r = json.loads(f.read_text())
        for k, value in r["profile"].items():
            if abs(float(k) - r["boundary"]) < 1e-9:
                continue                                                     # the split at the experience is reported apart
            rows.append(f"{r['name']}\t{r['boundary']:.6f}\t{float(k):.2f}\t{value - r['one_law']:.4f}")
        rows.append(f"{r['name']}\t{r['boundary']:.6f}\tboundary\t{r['profile'][str(r['boundary'])] - r['one_law']:.4f}")
    (out / "timing_profiles.tsv").write_text("\n".join(rows) + "\n")


COPIES = ("e64_posthoc_states.tsv", "e65_summary.json", "e65_pairwise_fit.tsv", "e66_check.tsv", "e66_check_above.tsv", "e66_check_truth.tsv")


def copies(out, inputs):
    """Verification tables of the project that the figures read, copied unchanged."""
    for name in COPIES:
        source = PROJECT / "verification" / name
        if not source.is_file():
            source = PAPER / "evidence" / "figures" / name
        inputs.append(source)
        (out / name).write_bytes(source.read_bytes())


def pairwise_intervals(root, out, inputs):
    """Empirical percentiles of the same eligible simulated estimates as the study summary."""
    source = root / "e65" / "fit2b" / "cohorts.jsonl"
    inputs.append(source)
    with source.open() as handle:
        cohorts = [json.loads(line) for line in handle if line.strip()]
    if sorted(row["replicate"] for row in cohorts) != list(range(200)):
        raise ValueError("Expected each of the 200 simulated cohorts exactly once")
    summary = json.loads((out / "e65_summary.json").read_text())
    with (out / "e65_intervals.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("session", "term", "usable", "lower", "upper"))
        for session in [*sorted(summary["sessions"]), "matched"]:
            for term in ("pairs", "trios"):
                if session == "matched":
                    ref = summary["matched_sessions"][term]
                    values = [r[term]["matched"] for r in cohorts]
                else:
                    ref = summary["sessions"][session][term]
                    values = [r[term]["sessions"][session] for r in cohorts]
                estimates = [v[0] for v in values if v is not None and np.isfinite(v[:2]).all() and v[1] > 0]
                if len(estimates) != ref["usable"] or not np.allclose(
                        [np.mean(estimates), np.std(estimates, ddof=1)], [ref["mean"], ref["sd"]], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Mismatched reference distribution for {session}, {term}")
                bounds = np.quantile(estimates, [0.025, 0.975]) if len(estimates) >= 20 else ("", "")
                writer.writerow((session, term, len(estimates), *bounds))


def revision():
    root = PROJECT.parent if (PROJECT.parent / "MANIFEST.json").is_file() else PROJECT.parents[1]
    manifest = root / "MANIFEST.json"
    if manifest.is_file():
        return json.loads(manifest.read_text())["source_commit"]
    if (root / ".git").exists():
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    inputs = []
    known_edge(a.results, a.out, inputs)
    hippocampus(a.results, a.out, inputs)
    timing(a.results, a.out, inputs)
    copies(a.out, inputs)
    pairwise_intervals(a.results, a.out, inputs)
    outputs = ("known_edge_examples.tsv", "hippocampus_examples.json", "timing_profiles.tsv", "e65_intervals.tsv", *COPIES)
    sources = [Path(__file__).resolve(), PROJECT / "real_data/dryad_posthoc_content.py", PROJECT / "real_data/replay_sequences.py"]
    def input_name(path):
        return str(path.relative_to(a.results)) if path.is_relative_to(a.results) else "verification/" + path.name
    receipt = dict(code_revision=revision(), script=Path(__file__).name,
                   code_sha256={path.name: sha(path) for path in sources},
                   inputs={input_name(path): sha(path) for path in inputs},
                   outputs={name: sha(a.out / name) for name in sorted(outputs)})
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: len(v) if isinstance(v, dict) else v for k, v in receipt.items()}))


if __name__ == "__main__":
    main()
