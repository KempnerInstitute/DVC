"""Write tables/tab_hippocampus.tex from the committed tables of the hippocampal study; nothing is estimated here.

    python make_hippocampus_table.py tables/tab_hippocampus.tex

Rows and standard errors are those of `draw_fig4_hippocampus.py` (its header says which standard error a row carries),
with Buddy added for the pair term.
"""
import csv
import sys
from pathlib import Path

ROWS = [  # label, table, group, column of the standard error, three-way term reported
    ("\\emph{First cohort}", None, None, None, None),
    ("Achilles (2 sessions)", "first_cohort_reserved.tsv", "achilles", "se_common_factor", True),
    ("Cicero (3)", "first_cohort_reserved.tsv", "cicero", "se_common_factor", True),
    ("Buddy (1)", "first_cohort_reserved.tsv", "buddy", "se_common_factor", False),
    ("Achilles and Cicero", "first_cohort_reserved.tsv", "achilles_cicero", "se_common_factor", True),
    ("Gatsby, development (2)", "first_cohort_development.tsv", "gatsby", "se_calibrated", True),
    ("\\emph{Second cohort}", None, None, None, None),
    ("Rat 1 (1 session)", "second_cohort.tsv", "rat1", "se_rule", True),
    ("Rat 2 (2)", "second_cohort.tsv", "rat2", "se_rule", True),
    ("Rat 3 (2)", "second_cohort.tsv", "rat3", "se_rule", True),
    ("Rat 4 (2)", "second_cohort.tsv", "rat4", "se_rule", True),
    ("Rat 5 (2)", "second_cohort.tsv", "rat5", "se_rule", True),
    ("Five rats (9)", "second_cohort.tsv", "cohort", "se_rule", True),
]


def read(folder, name):
    path = folder / name
    if not path.exists():
        return {}
    with path.open() as handle:
        return {(r["group"], r["contrast"], r["quantity"]): r for r in csv.DictReader(handle, delimiter="\t")}


def signed(value, digits=3):
    return f"${value:+.{digits}f}$".replace("-", "-")


def build(paper):
    folder = paper / "evidence" / "hippocampus"
    tables = {name: read(folder, name) for name in {row[1] for row in ROWS if row[1]}}
    lines = ["\\begin{tabular}{lrrrrr}\\toprule",
             " & \\multicolumn{2}{c}{Overlapping} & Pair term & \\multicolumn{2}{c}{Three-way term} \\\\",
             "Animal & pairs & trios & difference (SE) & difference (SE) & 95\\% interval \\\\\\midrule"]
    for label, table, group, column, three in ROWS:
        if table is None:
            lines.append(f"{label} & & & & & \\\\")
            continue
        get = lambda quantity: tables[table].get((group, "maze", quantity))
        pairs, trios = get("pairs_difference"), get("trios_difference")
        if pairs is None:
            continue
        cells = [label, f"{int(get('pairs_overlapping')['sets']):,}", f"{int(get('trios_overlapping')['sets']):,}",
                 f"{signed(float(pairs['delta']))} ({abs(float(pairs[column])):.3f})"]
        if three and trios is not None and trios["delta"]:
            estimate, se = float(trios["delta"]), abs(float(trios[column]))
            cells += [f"{signed(estimate)} ({se:.3f})", f"{signed(estimate - 1.96 * se)} to {signed(estimate + 1.96 * se)}"]
        else:
            cells += ["not used", ""]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.write_text(build(Path(__file__).resolve().parent))
    print(out.read_text())
