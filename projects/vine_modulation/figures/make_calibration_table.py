"""Typeset the class-change calibration counts from their simulation summaries."""
import argparse
import csv
from pathlib import Path


def render(folder):
    def counts(file, key):
        with (folder / file).open() as stream:
            rows = [r for r in csv.DictReader(stream,delimiter="\t") if r["comparison"] == "class_change"]
        result = {(r["group"],r["scenario"]):int(r[key]) for r in rows}
        assert len(rows) == len(result) == 6 and all(int(r["cohorts"]) == 24 for r in rows)
        return result
    standard = counts("fold_calibration.tsv","training_scaled_exceedances")
    common = counts("common_count_scenarios.tsv","crossings")
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Matched to & Count unit & Replay alone & Shared change & Overlap change \\", r"\midrule"]
    for group, label in (("cohort","Independent cohort"),("achilles","Achilles")):
        for unit, data in (("Per class SD",standard),("Per active pair",common)):
            cells = " & ".join(str(data[group,condition]) for condition in ("replay","apart","class"))
            lines.append(f"{label} & {unit} & {cells} " + r"\\")
    lines.extend((r"\bottomrule",r"\end{tabular}"))
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent/"evidence"/"hippocampus")
    args = parser.parse_args()
    args.output.write_text(render(args.inputs))
