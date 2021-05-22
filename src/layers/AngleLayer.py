import torch
from torch.nn import Parameter, Module
from torch.autograd import Variable
import numpy as np


class AngleLayer(Module):
    """Convert the fully connected layer of output to """
    def __init__(self, in_planes, out_planes, m=4):
        super(AngleLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.cos_val = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x,
        ]

    def forward(self, input):
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = torch.linalg.norm(input, dim=1)
        w_modulus = torch.linalg.norm(w, dim=1)

        # W * x = ||W|| * ||x|| * cos(θ)
        inner_wx = input.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1) / w_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_m_theta = self.cos_val[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        # k * pi / m <= theta <= (k + 1) * pi / m
        k = (self.m * theta / np.pi).floor()
        minus_one = k * 0.0 - 1
        # Phi(yi, i) = (-1)**k * cos(myi,i) - 2 * k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k

        cos_x = cos_theta * x_modulus.view(-1, 1)
        phi_x = phi_theta * x_modulus.view(-1, 1)

        return cos_x, phi_x
