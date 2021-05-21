# Copyright 2021 Massachusetts Institute of Technology
#
# @file visualization
# @author W. Nicholas Greene
# @date 2020-01-10 19:22:35 (Fri)

import numpy as np

import matplotlib
import matplotlib.cm

import torch

def pyramid_spiral(image_pyr, dataformats="NCHW"):
    """Arrange a power-of-two image pyramid in a spiral.
    """
    if dataformats == "NCHW":
        row_idx = 2
        col_idx = 3
        batch = image_pyr[0].shape[0]
        channels = image_pyr[0].shape[1]
        rows = image_pyr[0].shape[row_idx]
        cols = image_pyr[0].shape[col_idx]
    elif dataformats == "NHWC":
        row_idx = 1
        col_idx = 2
        batch = image_pyr[0].shape[0]
        rows = image_pyr[0].shape[row_idx]
        cols = image_pyr[0].shape[col_idx]
        channels = image_pyr[0].shape[3]
    else:
        assert(False)

    new_cols = 3 * cols // 2

    if dataformats == "NCHW":
        ret = np.zeros((batch, channels, rows, new_cols), dtype=image_pyr[0].dtype)
    elif dataformats == "NHWC":
        ret = np.zeros((batch, rows, new_cols, channels), dtype=image_pyr[0].dtype)
    else:
        assert(False)

    curr_roi = [0, 0, cols, rows]
    for lvl in range(len(image_pyr)):
        if dataformats == "NCHW":
            ret[:, :, curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] + curr_roi[2]] = image_pyr[lvl]
        elif dataformats == "NHWC":
            ret[:, curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] + curr_roi[2], :] = image_pyr[lvl]
        else:
            assert(False)

        if (lvl % 4 == 0):
          # Place new image on the right/top.
          curr_roi[0] += image_pyr[lvl].shape[col_idx]
          curr_roi[2] = image_pyr[lvl].shape[col_idx] // 2
          curr_roi[3] = image_pyr[lvl].shape[row_idx] // 2
        elif (lvl % 4 == 1):
          # Place new image on the bottom/right.
          curr_roi[0] += image_pyr[lvl].shape[col_idx] // 2
          curr_roi[1] += image_pyr[lvl].shape[row_idx]
          curr_roi[2] = image_pyr[lvl].shape[col_idx] // 2
          curr_roi[3] = image_pyr[lvl].shape[row_idx] // 2
        elif (lvl % 4 == 2):
          # Place new image on the left/bottom.
          curr_roi[0] -= image_pyr[lvl].shape[col_idx] // 2
          curr_roi[1] += image_pyr[lvl].shape[row_idx] // 2
          curr_roi[2] = image_pyr[lvl].shape[col_idx] // 2
          curr_roi[3] = image_pyr[lvl].shape[row_idx] // 2
        elif (lvl % 4 == 3):
          # Place new image on the top/left.
          curr_roi[1] -= image_pyr[lvl].shape[row_idx] // 2
          curr_roi[2] = image_pyr[lvl].shape[col_idx] // 2
          curr_roi[3] = image_pyr[lvl].shape[row_idx] // 2

    return ret;

def apply_normal_map(normals):
    """Colormap a set of normal vectors.
    """
    normals_cpu = normals.detach().squeeze().cpu()

    rgb = normals_cpu
    rgb[:, 0, :, :] += 1
    rgb[:, 0, :, :] *= 0.5 * 255

    rgb[:, 1, :, :] += 1
    rgb[:, 1, :, :] *= 0.5 * 255

    rgb[:, 2, :, :] *= 255/2
    rgb[:, 2, :, :] += 255/2

    return rgb.numpy().astype(np.uint8)

def apply_cmap(value, vmin=None, vmax=None, cmap=None):
    """A utility function that maps a grayscale image with a matplotlib colormap for
    use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 per batch
    item.

    Arguments:
      - value: 4D Tensor of shape [batch, 1, height, width].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`

    Returns a 4D RGBA numpy array of shape [batch, height, width, 4].
    """
    assert(len(value.shape) == 4)

    value_cpu = value.detach().cpu()
    value_cpu = torch.squeeze(value_cpu, dim=1)

    batch_size = value_cpu.shape[0]
    rows = value_cpu.shape[1]
    cols = value_cpu.shape[2]

    if vmin is None:
        vmin, _ = torch.min(value_cpu.view(batch_size, -1), 1, keepdim=True)
        vmin = vmin.unsqueeze(2).expand(-1, rows, cols)

    if vmax is None:
        vmax, _ = torch.max(value_cpu.view(batch_size, -1), 1, keepdim=True)
        vmax = vmax.unsqueeze(2).expand(-1, rows, cols)

    normalized_value = (value_cpu - vmin) / (vmax - vmin)

    if cmap is None:
        cmap = matplotlib.cm.get_cmap("gray")

    mapped = cmap(normalized_value.numpy())

    return mapped
