# src/sampling/vine_sample.py
import torch
from classes.objects import vine_obj_bin

def vine_copula_sample(vine: vine_obj_bin, n_samples: int):
    """
    Sample from the fitted vine using its sample() method.
    """
    return vine.sample((n_samples,))