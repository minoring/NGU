from torch.nn import init


def weight_init(layer, gain=init.calculate_gain('relu'), bias_const=0.):
    """Weight initializer nn.Module layers."""
    init.orthogonal_(layer.weigth, gain)
    layer.bias.data.fill_(bias_const)

    return layer
