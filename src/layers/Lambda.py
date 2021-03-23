from torch.nn import Module

__all__ = ['Lambda']

class Lambda(Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
