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


def transpose_batch(tensors):
    res = []
    for t in tensors:
        res.append(torch.transpose(t, 1, 0))
    return res


def make_one_hot(idx, num_class):
    """Make one-hot vector at idx filled with value."""
    one_hot = torch.zeros((num_class, ))
    one_hot[idx] = 1.0
    return one_hot


def make_one_hot_batch(idxs, num_class):
    batch_size = idxs.shape[0]
    one_hot = torch.zeros((batch_size, num_class), device=idxs.device)
    one_hot[:, idxs] = 1.0
    return one_hot
