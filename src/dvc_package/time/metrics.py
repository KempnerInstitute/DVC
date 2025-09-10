import numpy as np
from typing import List


def cusum_change_points(series: List[float], threshold: float = 1.0) -> List[int]:
    """Simple CUSUM change-point detection on a scalar time series.

    Args:
        series: sequence of numbers
        threshold: detection threshold
    Returns:
        indices of detected change points (approximate)
    """
    s_pos = 0.0
    s_neg = 0.0
    changes = []
    mean = np.nanmean(series)
    for i, x in enumerate(series):
        if not np.isfinite(x):
            continue
        d = x - mean
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_pos > threshold:
            changes.append(i)
            s_pos = 0.0
            s_neg = 0.0
        elif -s_neg > threshold:
            changes.append(i)
            s_pos = 0.0
            s_neg = 0.0
    return changes


def edge_persistence(adj_mats: List[np.ndarray]) -> float:
    """Compute edge persistence across time as average Jaccard similarity.

    Args:
        adj_mats: list of binary adjacency matrices over time
    Returns:
        mean Jaccard similarity across consecutive time steps
    """
    js = []
    for i in range(len(adj_mats) - 1):
        A = adj_mats[i].astype(bool)
        B = adj_mats[i + 1].astype(bool)
        inter = np.logical_and(A, B).sum()
        union = np.logical_or(A, B).sum()
        if union == 0:
            js.append(1.0)
        else:
            js.append(inter / union)
    return float(np.mean(js)) if js else float('nan')


