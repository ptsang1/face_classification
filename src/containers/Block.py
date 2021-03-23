import torch.nn as nn
from .ResidualBlock import ResidualBlock

__all__ = ['Block']


class Block(nn.Module):
    def __init__(self, layer_list):
        super(Block, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.traces = []

    def forward(self, x):
        for layer in self.layer_list:
            if isinstance(layer, ResidualBlock):
                x = layer(x, self.traces[layer.trace - 1])
            else:
                x = layer(x)
            self.traces.append(x)
        self.traces.clear()
        return x
