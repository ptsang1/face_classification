import torch.nn as nn

__all__ = ['ResidualBlock']


class ResidualBlock(nn.Module):
    def __init__(self, shortcut, block=None):
        super(ResidualBlock, self).__init__()
        self.shortcut = nn.Sequential(*shortcut)
        self.trace = block

    def forward(self, x, identity):
        return self.shortcut(identity) + x
