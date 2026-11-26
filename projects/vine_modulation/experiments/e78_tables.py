"""E78: small verification tables from the outputs of `real_data/shared_factor_content.py`.

    e78_tables.py RESULTS_ROOT VERIFICATION_FOLDER

Reads RESULTS_ROOT/e78/{dryad_shared_v1, posthoc_shared_v1}/{fit_comparison.json, study.json,
content_only_summary.json} and the separate-factor results of E66 and E68
(RESULTS_ROOT/hippocampus/{dryad_content_v1, posthoc_content_v1}/content_only_summary.json), and writes
  e78_fit.tsv       per session or hemisphere: in-sample log likelihood of the separate and the shared-factor laws,
                    their gap, AIC and BIC differences, the movement of the separate factors between rests, and the
                    share of ripples on a stretch of at most about 1 m before and after under both laws;
  e78_content.tsv   per group and estimate: recorded value (SE) and the cohorts of the shared-factor laws (mean,
                    spread, central 95%, cohorts at or above the recorded value), beside the same numbers for the
                    separate-factor laws of E66/E68;
  e78_summary.json  the totals of the fit comparison and the readings by the rule of experiments/E78_SHARED_FACTORS.md.
"""
import json
import sys
from pathlib import Path

FOLDERS = (("second", "dryad_shared_v1", "dryad_content_v1", ("cohort", "rat1", "rat2", "rat3", "rat4", "rat5")),
           ("first", "posthoc_shared_v1", "posthoc_content_v1", ("achilles", "cicero", "achilles_cicero", "gatsby")))


def main(root, verification):
    root, verification = Path(root), Path(verification)
    fit_rows = ["cohort\tunit\tplace_cells\tevents\tll_separate\tll_shared\tgap\tgap_per_event\taic_difference\tbic_difference\tcycles\t"
                "factor_move_median_abs\tfactor_move_mean\tfactor_move_share_up\tstretch_pre_separate\tstretch_post_separate\tstretch_pre_shared\tstretch_post_shared"]
    content_rows = ["cohort\tgroup\testimate\tterm\trecorded\trecorded_se\tshared_mean\tshared_sd\tshared_low\tshared_high\tshared_at_or_above\tshared_usable\t"
                    "separate_mean\tseparate_low\tseparate_high\tseparate_at_or_above"]
    summary = {}
    for cohort, shared_name, separate_name, groups in FOLDERS:
        shared, separate = root / "e78" / shared_name, root / "hippocampus" / separate_name
        fits = json.loads((shared / "fit_comparison.json").read_text())
        for unit, row in sorted(fits["units"].items()):
            r = row["rests"]
            fit_rows.append("\t".join(str(v) for v in (cohort, unit, row["place_cells"], row["events"], f"{row['separate']:.3f}", f"{row['shared']:.3f}",
                                                       f"{row['gap']:.3f}", f"{row['gap_per_event']:.5f}", f"{row['aic_difference']:.3f}",
                                                       f"{row['bic_difference']:.3f}", row["cycles"], f"{row['factor_move']['median_abs']:.4f}",
                                                       f"{row['factor_move']['mean']:.4f}", f"{row['factor_move']['share_up']:.3f}",
                                                       f"{r['PRE']['stretch_share_separate']:.4f}", f"{r['POST']['stretch_share_separate']:.4f}",
                                                       f"{r['PRE']['stretch_share_shared']:.4f}", f"{r['POST']['stretch_share_shared']:.4f}")))
        simulated = json.loads((shared / "content_only_summary.json").read_text())
        before = json.loads((separate / "content_only_summary.json").read_text())
        for group in groups:
            for name, v in simulated["groups"].get(group, {}).items():
                estimate, term = name.split("_", 1)
                w = before["groups"].get(group, {}).get(name, {})
                content_rows.append("\t".join(str(x) for x in (cohort, group, estimate, term, f"{v['recorded']:.4f}", f"{v['recorded_se']:.4f}", f"{v['mean']:.4f}",
                                                               f"{v['sd']:.4f}", f"{v['low']:.4f}", f"{v['high']:.4f}", v["cohorts_at_or_above_recorded"],
                                                               v["usable"], f"{w.get('mean', float('nan')):.4f}", f"{w.get('low', float('nan')):.4f}",
                                                               f"{w.get('high', float('nan')):.4f}", w.get("cohorts_at_or_above_recorded", ""))))
        summary[cohort] = dict(fit=fits["summary"], cohorts=simulated["cohorts"], reading=simulated.get("reading"),
                               separate_reading=before.get("reading"))
    verification.mkdir(parents=True, exist_ok=True)
    (verification / "e78_fit.tsv").write_text("\n".join(fit_rows) + "\n")
    (verification / "e78_content.tsv").write_text("\n".join(content_rows) + "\n")
    (verification / "e78_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
