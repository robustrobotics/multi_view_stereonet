# Copyright 2021 Massachusetts Institute of Technology
#
# @file image_utils.py
# @author W. Nicholas Greene
# @date 2019-12-07 16:58:41 (Sat)

import numpy as np

import torch

def GaussianBlur(device, kernel_size=5, sigma=1.0, channels=1):
    """Build a Gaussian lowpass filter.

    Taken from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/2
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * np.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    blur = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                           kernel_size=kernel_size, groups=channels, bias=False,
                           padding=kernel_size//2, padding_mode="border")

    blur.weight.data = gaussian_kernel.to(device)
    blur.weight.requires_grad = False

    return blur

def blur_with_zeros(image, blur):
    """Blur an image ignoring zeros.
    """
    # Create binary weight image with 1s where valid data exists.
    mask = torch.ones(image.shape, device=image.device)
    mask[image <= 0] = 0

    # Compute blurred image that ignores zeros using ratio of blurred images.
    blurred = blur(image)
    weights = blur(mask)

    blurred[weights == 0] = 0
    weights[weights == 0] = 1
    blurred = blurred / weights

    return blurred

def forward_gradx(image):
    """Compute forward horizontal gradient.

    Image will be padded by 1 element on the right.

    Assumes input is a 4D tensor with dimensions (batch, channels, rows, cols).
    """
    assert(len(image.shape) == 4)
    image_pad = torch.nn.functional.pad(image, (0, 1, 0, 0), mode="replicate")
    gx = image_pad[:, :, :, :-1] - image_pad[:, :, :, 1:]
    return gx

def forward_grady(image):
    """Compute forward vertical gradient.

    Image will be padded by 1 element on the bottom.

    Assumes input is a 4D tensor with dimensions (batch, channels, rows, cols).
    """
    assert(len(image.shape) == 4)
    image_pad = torch.nn.functional.pad(image, (0, 0, 0, 1), mode="replicate")
    gy = image_pad[:, :, :-1, :] - image_pad[:, :, 1:, :]
    return gy

def central_gradx(image):
    """Compute central horizontal gradient.

    Assumes input is a 4D tensor with dimensions (batch, channels, rows, cols).
    """
    assert(len(image.shape) == 4)
    padder = torch.nn.ReplicationPad2d((1, 1, 0, 0))
    padded_image = padder(image)
    gx = 0.5 * (padded_image[:, :, :, 2:] - padded_image[:, :, :, :-2])
    return gx

def central_grady(image):
    """Compute central vertical gradient.

    Assumes input is a 4D tensor with dimensions (batch, channels, rows, cols).
    """
    assert(len(image.shape) == 4)
    padder = torch.nn.ReplicationPad2d((0, 0, 1, 1))
    padded_image = padder(image)
    gy = 0.5 * (padded_image[:, :, 2:, :] - padded_image[:, :, :-2, :])
    return gy

def build_image_pyramid(image, num_levels):
    """Create an image pyramid.

    Derived from monodepth.
    """
    assert(len(image.shape) == 4)

    pyramid = [image]
    for lvl in range(1, num_levels):
        hlvl = (pyramid[-1].shape[2] + 1) // 2
        wlvl = (pyramid[-1].shape[3] + 1) // 2

        shape = (hlvl, wlvl)

        image_lvl = torch.nn.functional.interpolate(pyramid[-1], shape, mode="area")
        pyramid.append(image_lvl)

    return pyramid
