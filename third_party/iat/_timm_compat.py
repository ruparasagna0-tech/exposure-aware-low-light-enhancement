"""[phase5 patch] Minimal stand-ins for the three timm helpers IAT uses, so IAT runs without timm.

trunc_normal_ : torch.nn.init.trunc_normal_ (same signature/semantics as timm's: tensor, mean, std, a, b)
DropPath      : stochastic depth; identity in eval() (all our use is inference)
to_2tuple     : int -> (int, int)
"""
import collections.abc

import torch
import torch.nn as nn

trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(x):
    return tuple(x) if isinstance(x, collections.abc.Iterable) else (x, x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep
