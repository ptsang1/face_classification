import torch
import torch.nn as nn
from containers.ResidualBlock import ResidualBlock
from layers.Lambda import Lambda
from .helper import *
from .AngleLayer import AngleLayer


class CreateLayer:
    def __init__(self):
        self.layers = {
            'conv2': lambda input_shape, **kwargs: conv2(input_shape, **kwargs),
            'linear': lambda input_shape, **kwargs: linear(input_shape, **kwargs),
            'flatten': lambda input_shape: (
                Lambda(lambda x: x.view(x.size(0), -1)), torch.Size([1, input_shape.numel()])),
            'trace': lambda input_shape, **kwargs: self.trace(input_shape, **kwargs),
            'batch_norm2': lambda input_shape: (nn.BatchNorm2d(input_shape[0]), input_shape),
            'max_pooling': lambda input_shape, **kwargs: max_pooling(input_shape, **kwargs),
            'identity': lambda: Lambda(lambda x: x),
            'angle': lambda preceding_units, n_classes, margin=4: AngleLayer(preceding_units, n_classes, m=4),
            'relu': lambda input_shape, inplace=False: (nn.ReLU(inplace), input_shape),
            'prelu': lambda input_shape, inplace=False: (nn.PReLU(inplace), input_shape),
            'resnet_basic_block': lambda input_shape, **kwargs: (
                nn.Sequential(nn.Conv2d(input_shape[0], input_shape[0], kernel_size=3, padding=1),
                              nn.BatchNorm2d(input_shape[0]),
                              nn.PReLU(),
                              nn.Conv2d(input_shape[0], input_shape[0], kernel_size=3, padding=1),
                              nn.BatchNorm2d(input_shape[0]),
                              nn.PReLU()
                              ), input_shape)
        }

    def create_layer(self, layer, input_shape):
        operations = []
        if len(layer) == 1:
            name = layer[0].pop('name')
            return self.layers[name](input_shape, **layer[0])
        for params in layer:
            name = params.pop('name')
            operation, output_shape = self.layers[name](input_shape, **params)
            operations.append(operation)
            input_shape = output_shape
        return nn.Sequential(*operations), input_shape

    def create_layers(self, input_shape, model_json=None):
        converted_model = []
        if not input_shape:
            return converted_model
        if model_json is not None:
            for layer in model_json:
                block, output_shape = self.create_layer(layer, input_shape)
                converted_model.append(block)
                input_shape = output_shape
            return converted_model, input_shape
        return converted_model

    def trace(self, input_shape, **kwargs):
        block = kwargs.get("block")
        shortcut, output_shape = self.create_layer(kwargs.get("shortcut", [{"name": "identity"}]), input_shape)
        return ResidualBlock(shortcut, block), output_shape
