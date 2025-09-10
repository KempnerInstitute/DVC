# src/info/info_estimation.py
import torch
from sampling.vine_sample import vine_copula_sample

def vine_entropy(vine, info_dict: dict):
    """
    Estimate the differential entropy H = -E[log f(X)] via Monte Carlo.
    """
    cases = info_dict.get('cases', 1000)
    samples = vine_copula_sample(vine, cases)
    _, _, logf = vine.evaluation(samples)
    H_est = - torch.mean(logf)
    return H_est.item()