# Contributing to DVC

Keep estimators and shared statistical routines in `src/dvc_package/`, runnable examples in `examples/`, and configuration-driven tools in `scripts/`. A study under `projects/` contains its protocols, figure scripts, compact figure inputs and reproduction guide. Document changes to public methods in `docs/reference/` or `docs/user-guide/`.

For a behavioral change, include a focused test and state the inputs and expected result in the pull request. Preserve the original source citations and random seeds when updating a published analysis.

## Check a change

From the repository root, install the library and development tools in a Python 3.10 or 3.11 environment.

```bash
python -m pip install -e . pytest mkdocs
python scripts/test_installation.py
python -m pytest -q tests
mkdocs build --strict
```

Changes to a paper project should also pass the tests and worked examples in its study guide. Submit only the source, documentation and compact reference data needed to reproduce the change. Keep credentials, raw recordings, environment files and machine-specific settings outside Git.
