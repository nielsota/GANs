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


# Code for zipping together lists of unequal length
def _one_pass(iters):
    for it in iters:
        try:
            yield next(it)
        except StopIteration:
            pass #of some of them are already exhausted then ignore it.


def zip_varlen(*iterables):
    iters = [iter(it) for it in iterables]
    while True: #broken when an empty tuple is given by _one_pass
        val = tuple(_one_pass(iters))
        if val:
            yield val
        else:
            break
