#!/usr/bin/env python3
"""Compatibility entrypoint for the final Figure 7 generator.

The paper build imports this module, while the reviewed benchmark workflow now
lives in ``scripts_ale_final``. Keep this thin wrapper so older commands still
produce the final showcase figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_ale_final.generate_fig7_showcase import main  # noqa: E402


if __name__ == "__main__":
    main()
