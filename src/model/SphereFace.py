import warnings
import torch.nn as nn

from containers.Block import Block
from layers.AngleLayer import AngleLayer
from layers.CreateLayer import CreateLayer

warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ['SphereFace']


class SphereFace(nn.Module):
    def __init__(self, model, n_classes, input_shape, **kwargs):
        """
        :param model: model json file
        :param n_classes: the number of classes
        :param input_shape: the shape of input
        :param kwargs:
        """
        super(SphereFace, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        converted_model, output_shape = CreateLayer().create_layers(self.input_shape, model)
        self.model = Block(converted_model)
        self.angleLayer = AngleLayer(output_shape[1], n_classes)

    def forward(self, x):
        x = self.model(x)
        y = self.angleLayer(x)
        return x, y
