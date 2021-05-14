import torch.nn as nn

__all__ = ['ResidualBlock']


class ResidualBlock(nn.Module):
    def __init__(self, shortcut=[], block=None):
        super(ResidualBlock, self).__init__()
        if shortcut and len(shortcut) > 0:
            self.shortcut = nn.Sequential(*shortcut)
        else:
            self.shortcut = None
        self.trace = block

    def forward(self, x, identity):
        if self.shortcut:
            return self.shortcut(identity) + x
        return x + identity
