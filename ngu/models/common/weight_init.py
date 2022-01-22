import torch.nn as nn
from torch.nn import init


def weight_init(layer, gain=init.calculate_gain('relu'), bias_const=0.):
    """Weight initializer nn.Module layers."""
    if isinstance(layer, nn.LSTMCell):
        # https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
        for name, param in layer.named_parameters():
            if 'bias' in name:
                param.data.fill_(bias_const)
            elif 'weight' in name:
                init.orthogonal_(param, gain)
    else:
        init.orthogonal_(layer.weight, gain)
        layer.bias.data.fill_(bias_const)

    return layer
