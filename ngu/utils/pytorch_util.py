import torch

device = None


def init_device(use_gpu=True):
    global device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('No GPU. defaults to use CPU.')


def to_tensor(arr):
    return torch.from_numpy(arr).float().to(device)


def to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()
