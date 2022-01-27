import torch
import numpy as np

device = None


def init_device(use_gpu=True):
    global device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('No GPU. defaults to use CPU.')


def to_tensor(x):
    if isinstance(x, np.ndarray):
        # Case where it is numpy array.
        return torch.from_numpy(x).float().to(device)
    # Case where it is python scalar.
    return torch.tensor(x).float().to(device)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def to_list(tensor):
    return to_numpy(tensor).tolist()


def to_device(tensors, device):
    res = []
    for t in tensors:
        res.append(t.to(device))
    return res


def make_one_hot(val, idx, num_class):
    """Make one-hot vector at idx filled with value."""
    one_hot = torch.zeros((num_class, ))
    one_hot[idx] = val
    return one_hot
