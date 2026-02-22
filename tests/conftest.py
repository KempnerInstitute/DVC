import random
import sys
from pathlib import Path

import numpy as np
import torch

import pytest

# Ensure `src/` is importable for local test runs without editable install.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
