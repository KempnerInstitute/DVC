# Data Release Plan

DVC uses synthetic data, public datasets, and derived paper artifacts. The code
release should make each category explicit.

## Synthetic Benchmarks

Release benchmark generators as code and record:

- scenario name,
- dimension and sample counts,
- seed and split convention,
- copula families and parameters,
- output metric units.

Generated synthetic result tables can be included as small CSV/JSON artifacts
with checksums.

## Public Real Datasets

For public source datasets, do not bundle raw data unless redistribution is
explicitly permitted. Instead provide:

- source archive URL or DOI,
- citation,
- license or terms of use,
- download script where allowed,
- preprocessing script,
- derived-feature schema,
- checksum for each derived file.

## Paper Artifact Bundle

Recommended structure for a release artifact:

```text
dvc-paper-artifacts/
├── README.md
├── MANIFEST.json
├── configs/
├── derived_tables/
├── figures/
├── result_summaries/
└── checksums.sha256
```

Host large derived files on Hugging Face Datasets or an archival repository and
link the DOI or dataset URL from the paper and the GitHub release.

## Privacy and Ethics

The current real-data analyses use public animal-neuroscience datasets and do
not include newly collected human-subject data. Still, the release should avoid
including local paths, machine-specific logs, unpublished notebooks, or exploratory
notes that are not necessary for reproducibility.
