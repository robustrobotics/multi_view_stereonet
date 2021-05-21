# Copyright 2021 Massachusetts Institute of Technology
#
# @file pytorch_utils.py
# @author W. Nicholas Greene
# @date 2019-12-06 15:48:43 (Fri)

import time
import random
import os

import numpy as np

import torch

def set_seeds(seed, cudnn_deterministic):
    """Set all random number seeds.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if cudnn_deterministic:
            torch.backends.cudnn.deterministic = True

    return

def start_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        tick = torch.cuda.Event(enable_timing=True)
        tock = torch.cuda.Event(enable_timing=True)
        tick.record()
    else:
        tick = time.time()
        tock = None
    return tick, tock

def stop_timer(tick, tock):
    if torch.cuda.is_available():
        tock.record()
        torch.cuda.synchronize()
        return tick.elapsed_time(tock)
    else:
        return (time.time() - tick) * 1000

def num_parameters_conv2d(in_channels, out_channels, kernel_size, bias=True):
    """Compute number of parameters for a Conv2D layer.
    """
    if bias:
        return (kernel_size * kernel_size * in_channels + 1) * out_channels
    else:
        return (kernel_size * kernel_size * in_channels + 0) * out_channels

def num_trainable_parameters(model):
    """Return the number of trainable parameters for a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
