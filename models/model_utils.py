import torch
from torch import nn


class AddDimension(nn.Module):
    """
    Turn [B, T] -> [B, 1, T]
    """

    def forward(self, x):
        return x.view(len(x), 1, -1)


class SqeezeDimension(nn.Module):
    """
    Turn [B, C, T] -> [B, C*T]
    """

    def forward(self, x):
        return x.view(len(x), -1)


# Helper function for building hook
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook