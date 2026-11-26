"""Export the plotted values with units and replication levels in separate workbook sheets."""
import argparse
import csv
import json
from datetime import datetime
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from openpyxl.xml.functions import tostring
from statistics import mean, median, stdev
from scipy.stats import t as student_t
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from known_edge_timing import paired_intervals

DATA = Path(__file__).resolve().parent / "evidence"


def save_workbook(book, output):
    """Pin OOXML metadata and ZIP times so independent builds have the same digest."""
    buffer = BytesIO()
    book.save(buffer)
    epoch = datetime(1980, 1, 1)
    book.properties.created = epoch
    book.properties.modified = epoch
    with ZipFile(buffer) as source, ZipFile(output, "w") as target:
        for name in sorted(source.namelist()):
            data = tostring(book.properties.to_tree()) if name == "docProps/core.xml" else source.read(name)
            member = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o600 << 16
            member.compress_type = ZIP_DEFLATED
            target.writestr(member, data)


def export(output):
    book = Workbook()
    guide = book.active
    guide.title = "Guide"
    guide.append(("Sheet", "What the values measure", "Observation or denominator"))
    expected = {}
    def sheet(name, purpose, unit, rows):
        rows = list(rows)
        cols = tuple(rows[0])
        ws = book.create_sheet(name)
        ws.append(cols)
        for row in rows:
            ws.append([row[c] if row[c] != "" else None for c in cols])
        guide.append((name,purpose,unit))
        expected[name] = len(rows)
    def read_table(path, delimiter):
        with path.open() as source:
            rows = list(csv.DictReader(source, delimiter=delimiter))
        for row in rows:
            for key,value in row.items():
                # Spike patterns and row identifiers are labels. A leading zero records a silent cell.
                if key in ("law_or_pair", "row", "called_windows"):
                    continue
                try:
                    row[key] = int(value)
                except ValueError:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
        return rows

    worked = json.loads((DATA/"worked.json").read_text())
    figures = DATA/"figures"
    for number, name, purpose, unit in (
            (1, "Fig1 concept", "Coefficient paths of the moving-baseline law and Kendall tau of the simulated clouds", "Window, or one simulated cloud"),
            (2, "Fig2 worked", "True and fitted conditional Kendall tau, pairwise tau of the reversal simulation, binary patterns", "Window, pair or pattern"),
            (3, "Fig3 recovery", "Example outcomes, exact recovery by design and size, search recovery and specificity counts", "Estimator, design, law or check"),
            (4, "Fig4 hippocampus", "Example fields and patterns, and the overlap-minus-separate change of each animal", "Cell, pattern, animal or pooled cohort"),
            (5, "Fig5 checks", "Pairwise-model cohorts, sleep-state subsets and step-by-step changes", "Session, ripple subset or step"),
            (6, "Fig6 replay", "Decoded ripples, replay shares, recorded and simulated contrasts, simulated check", "Ripple, recording, estimate or scenario"),
            (7, "Fig7 whole ripples", "Held-out gains, calibration counts, timing profiles and schedule comparisons", "Recording, group, scenario or contrast")):
        sheet(name, purpose, unit, read_table(figures/f"fig{number}_source.csv", ","))
    sheet("Fig3 example paths", "True and fitted coefficient paths for the two illustrated simulations",
          "Design and window", read_table(figures/"known_edge_examples.tsv", "\t"))
    sheet("Fig5 simulation ranges", "Empirical pairwise-model percentiles; bounds require 20 valid estimates",
          "Session or pooled seven sessions, separately for pairs and trios", read_table(figures/"e65_intervals.tsv", "\t"))
    sheet("Fig7 replay profiles", "Replay split log likelihood relative to one fitted replay law",
          "Recording and candidate boundary, in nats", read_table(figures/"timing_profiles.tsv", "\t"))
    sheet("Fig3 timing intervals","95% simulation intervals for exact recovery and its paired difference",
          "Matched dataset resampled within its design and effect size",paired_intervals(DATA/"known_edge_timing.csv"))
    sheet("Fig3 known trials","Matched exact-recovery indicators","One of 720 simulated datasets",read_table(DATA/"known_edge_timing.csv",","))
    sheet("Fig3 search controls","Extra terms and held-out gains across eight simulation laws",
          "One frozen 50-variable dataset, independently matched to its receipt",read_table(DATA/"challenge_terms.tsv","\t"))
    plants = [dict(law=r["law"],n=r["n"],seed=r["seed"],plant_index=i,**t)
              for r in worked["challenge"] for i,t in enumerate(r["terms"],1)]
    sheet("Fig3 plants","All plants, including constant terms","Term within one simulated dataset",plants)
    hippo = DATA/"hippocampus"
    rats = [float(r["delta"]) for r in read_table(hippo/"second_cohort.tsv","\t")
            if str(r["group"]).startswith("rat") and r["contrast"]=="maze" and r["quantity"]=="trios_difference"]
    assert len(rats)==5
    m, se = mean(rats), stdev(rats)/len(rats)**.5
    width=student_t.ppf(.975,len(rats)-1)*se
    sheet("SI between animals","Equal-weight descriptive summary of third-order class changes in five rats",
          "Animal, approximate t interval with 4 degrees of freedom",
          [dict(rats=len(rats),mean=m,median=median(rats),standard_error=se,lower=m-width,upper=m+width)])
    comparisons = (
        ("fixed_fold_summary", "SI fixed boundary", "Fixed experimental boundary", "Cohort and model increment, in nats"),
        ("blind_fold_summary", "SI searched boundary", "Boundary searched on training data", "Cohort and model increment, in nats"),
        ("fixed_fold_recordings", "SI fixed recordings", "Fixed-boundary held-out log scores", "Recording and model increment, in nats"),
        ("blind_fold_recordings", "SI searched recordings", "Searched-boundary held-out log scores", "Recording and model increment, in nats"),
        ("timing_reference", "SI timing reference", "Estimated and known boundaries", "Recording; fraction of ordered ripples"),
        ("schedule_predictions", "SI schedule scores", "Held-out prediction against a fixed one-third split", "Recording or pooled comparison, in nats"),
        ("fold_calibration", "SI calibration", "Original and fold-specific class count units", "24 simulated cohorts per geometry and scenario"),
        ("fold_calibration_replicates", "SI calibration replicates", "Paired scores for each simulated cohort", "Replicate and nested model comparison"),
        ("common_count_scenarios", "SI per-pair calibration", "Change in log odds per additional active pair", "24 simulated cohorts per geometry and scenario"),
        ("common_count_populations", "SI per-pair prediction", "Increment in whole-ripple log score using a common count unit", "Cohort and model increment, in nats"),
        ("common_count_scale_reference", "SI count units", "Size of the planted shared shift in training-fold units", "Absolute discrepancy across recordings, replicates and folds"),
    )
    for filename, tab, description, unit in comparisons:
        sheet(tab, description, unit, read_table(hippo/(filename+".tsv"),"\t"))
    for ws in book:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col in ws.columns:
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(80, max(12,max(len(str(c.value or "")) for c in col)+2))
    save_workbook(book, output)
    with output.open("rb") as handle:
        checked = load_workbook(handle,read_only=True,data_only=True)
        assert all(checked[name].max_row == n+1 for name,n in expected.items())
        checked.close()
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.output),sort_keys=True))
