# Contributing

Thank you for improving Dynamic Vine Copulas. The project is organized as a
reusable scientific Python package with a separate paper/project workspace.

## Development Rules

- Keep reusable code under `src/dvc_package`.
- Keep public entry points under `scripts`.
- Keep paper-only scripts, generated artifacts, local runbooks, and exploratory
  notes under `drafts/projects`.
- Add focused tests for behavioral changes.
- Prefer deterministic seeds in tests and examples.
- Use logging in library code instead of print statements.
- Document public APIs in `docs/reference` and user-facing workflows in
  `docs/user-guide`.

## Before Opening a Pull Request

```bash
python scripts/test_installation.py
python -m pytest -q
mkdocs build
```

Also scan for local paths, machine-specific notes, generated data, and placeholder
metadata before publishing a branch.
