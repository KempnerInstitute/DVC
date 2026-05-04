# Release Plan

This checklist defines the path from the current research repository to a clean
public code release.

## Package Release

1. Keep reusable code in `src/dvc_package`.
2. Keep public entry points in `scripts/`.
3. Keep small examples in `examples/`.
4. Keep generated results, logs, raw data, and paper-only scripts out of the
   public package.
5. Run:

```bash
python scripts/test_installation.py
python -m pytest -q
mkdocs build
```

6. Tag a release after the tests and docs build pass from a fresh clone.

## Paper Reproduction Bundle

Prepare a separate artifact bundle for the paper with:

- exact configs and seeds,
- figure/table generation scripts,
- result manifests and checksums,
- small derived tables needed by the manuscript,
- instructions for downloading public source datasets,
- expected output filenames and units.

The bundle can be attached as anonymous supplementary material during review and
mirrored publicly after review.

## Public Data and Artifact Hosting

Use GitHub for code and small text artifacts. Use Hugging Face Datasets, Zenodo,
OSF, or another archival host for larger derived datasets and result bundles.
The release should avoid redistributing third-party raw data when the original
license or terms require users to download it from the source archive.

## Pre-Release Audit

- search for absolute local paths, machine-specific notes, placeholders, and stale action items
- `find . -name "__pycache__" -o -name "*.pyc"`
- `git status --short --ignored`
- inspect `README.md`, `pyproject.toml`, and `docs/` for placeholders.

Anything found by the audit should be removed, generalized, or moved under
`drafts/projects`.
