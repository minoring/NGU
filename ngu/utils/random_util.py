import random

import torch
import numpy as np


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
