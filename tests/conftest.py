import torch, numpy as np, random

import pytest

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234) 