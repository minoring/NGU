import torch


def conv2d_out(input_size, kernel_size, stride, padding=0):
    """Compute the output size of Conv2d"""
    return ((input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1)


def reward_transform(x, eps=0.001):
    """Instead of reward clipping, use an inverted value function rescaling"""
    return x.sign() * ((torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x)


def reward_transform_inverted(x, eps=0.001):
    """Inverted version of reward_transform."""
    return x.sign() * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) - 1)
