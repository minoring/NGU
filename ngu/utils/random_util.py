import random

import torch
import numpy as np


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def choose_among_with_prob(list_a, list_b, probs):
    """Select elements between two list where ith element of list_a element with 1 - probs[i], and
    ith element of list_b element with prob[i]"""
    res = []
    assert len(list_a) == len(list_b) == len(probs)
    for i, prob in enumerate(probs):
        if random.random() <= prob:
            res.append(list_b[i])
        else:
            res.append(list_a[i])
    return res
