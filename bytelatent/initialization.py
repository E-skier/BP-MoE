# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import nn
from torch.distributed._tensor import DTensor


def trunc_normal_(tensor: torch.Tensor, *args, **kwargs):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return nn.init.trunc_normal_(tensor, *args, **kwargs)
