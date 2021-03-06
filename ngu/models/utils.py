import torch


def conv2d_out(input_size, kernel_size, stride, padding=0):
    """Compute the output size of Conv2d"""
    return ((input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1)


def reward_transform(x, eps=0.001):
    """Instead of reward clipping, use an inverted value function rescaling"""
    return x.sign() * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def reward_transform_inverted(x, eps=0.001):
    """Inverted version of reward_transform."""
    return x.sign() * ((torch.sqrt(1.0 + 4.0 * eps * (x.abs() + 1.0 + eps)) - 1.0) /
                       (2.0 * eps) - 1.0)
