import torch
import torch.nn as nn


def conv2(input_shape, **kwargs):
    out_channels = kwargs.get('out_channels', input_shape[0])
    kwargs['out_channels'] = out_channels
    kernel_size = kwargs.get('kernel_size', 3)
    padding = kwargs.get('padding', 0)
    stride = kwargs.get('stride', 1)
    layer = nn.Conv2d(input_shape[0], bias=False, **kwargs)
    w = (input_shape[1] - kernel_size + 2 * padding) // stride + 1
    h = (input_shape[2] - kernel_size + 2 * padding) // stride + 1
    return layer, torch.Size([out_channels, w, h])


def linear(input_shape, **kwargs):
    out_features = kwargs.get('out_features', input_shape[0])
    layer = nn.Linear(input_shape[1], **kwargs)
    return layer, torch.Size([input_shape[0], out_features])


def max_pooling(input_shape, **kwargs):
    kernel_size = kwargs.get('kernel_size', 2)
    stride = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 0)
    w = (input_shape[1] - kernel_size + 2 * padding) // stride + 1
    h = (input_shape[2] - kernel_size + 2 * padding) // stride + 1
    return nn.MaxPool2d(**kwargs), torch.Size([input_shape[0], w, h])
