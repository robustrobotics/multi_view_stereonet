# Copyright 2021 Massachusetts Institute of Technology
#
# @file losses.py<stereo>
# @author W. Nicholas Greene
# @date 2019-12-07 16:57:08 (Sat)

import torch

from utils import image_utils

def pseudo_huber_loss(truth, pred, scale=2.0):
    """Pseudo-Huber loss used in StereoNet.

    Described in 2019 Barron - A General and Adaptive Robust Loss Function.
    """
    diff2 = ((pred - truth) / scale)**2
    loss = torch.mean(torch.sqrt(diff2 + 1.0) - 1.0)
    return loss

def corner_loss(features, patch_size):
    """Compute loss that promotes "corner"-ness.

    "Corner"-ness is some function of the eigenvalues of the Hessian of the
    photometric error function between two image patches.

    See: https://en.wikipedia.org/wiki/Corner_detection
    """
    assert(len(features.shape) == 4) # (batch, channel, rows, cols)

    # Normalize features. Otherwise loss is only valid up to scale.
    rows = features.shape[2]
    cols = features.shape[3]
    mu_features = torch.mean(features, dim=(2, 3), keepdim=True).expand(-1, -1, rows, cols)
    std_features = torch.std(features, dim=(2, 3), keepdim=True).expand(-1, -1, rows, cols)
    z_features = (features - mu_features) / (std_features + 1e-6)

    # Compute gradients.
    gx = image_utils.central_gradx(z_features)
    gy = image_utils.central_grady(z_features)

    gx2 = gx**2
    gy2 = gy**2
    gxy = gx * gy

    # Aggregate gradients over window using box filter.
    # Filter dims: (out_channels, in_channels, kernel_rows, kernel_cols)
    padding = patch_size // 2
    gx2_agg = torch.nn.functional.avg_pool2d(gx2, patch_size, stride=1, padding=padding)
    gy2_agg = torch.nn.functional.avg_pool2d(gy2, patch_size, stride=1, padding=padding)
    gxy_agg = torch.nn.functional.avg_pool2d(gxy, patch_size, stride=1, padding=padding)

    # Compute determinant of Hessian (product of eigenvalues). Other
    # functions can be used (e.g. harmonic mean, Harris, etc.)
    det = gx2_agg * gy2_agg - gxy_agg * gxy_agg

    mean_det = torch.mean(det)
    loss = torch.exp(-0.1 * mean_det)

    return loss

def gradient_matching_loss(image, features):
    """Compute loss that saves gradient information from the original images.
    """
    assert(len(image.shape) == 4) # (batch, channel, rows, cols)
    assert(len(features.shape) == 4) # (batch, channel, rows, cols)

    # Compute gradients. Remember to take mean over channel dimension.
    gx_image = torch.mean(image_utils.central_gradx(image), dim=1)
    gy_image = torch.mean(image_utils.central_grady(image), dim=1)

    mag_image = torch.sqrt(gx_image * gx_image + gy_image * gy_image)
    gxn_image = gx_image / (mag_image + 1e-3)
    gyn_image = gy_image / (mag_image + 1e-3)

    # Normalize features. Otherwise loss is only valid up to scale.
    rows = features.shape[2]
    cols = features.shape[3]
    mu_features = torch.mean(features, dim=(2, 3), keepdim=True).expand(-1, -1, rows, cols)
    std_features = torch.std(features, dim=(2, 3), keepdim=True).expand(-1, -1, rows, cols)
    z_features = (features - mu_features) / (std_features + 1e-6)

    gx_feature = torch.mean(image_utils.central_gradx(z_features), dim=1)
    gy_feature = torch.mean(image_utils.central_grady(z_features), dim=1)

    grad_proj = torch.mean(gxn_image * gx_feature + gyn_image * gy_feature)

    loss = torch.exp(-grad_proj)

    return loss

def SSIM(x, y, patch_size=3):
    """Structural similarity between two images x and y.

    Adapted from Monodepth.
    """
    assert(len(x.shape) == 4) # (batch, channel, rows, cols)
    assert(len(y.shape) == 4) # (batch, channel, rows, cols)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    padding = patch_size // 2

    mu_x = torch.nn.functional.avg_pool2d(x, patch_size, stride=1, padding=padding)
    mu_y = torch.nn.functional.avg_pool2d(y, patch_size, stride=1, padding=padding)

    sigma_x  = torch.nn.functional.avg_pool2d(x ** 2, patch_size, stride=1, padding=padding) - mu_x ** 2
    sigma_y  = torch.nn.functional.avg_pool2d(y ** 2, patch_size, stride=1, padding=padding) - mu_y ** 2
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, patch_size, stride=1, padding=padding) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim = ssim_n / ssim_d

    ssim = (1 - ssim) / 2
    ssim[ssim < 0] = 0
    ssim[ssim > 1] = 1

    return ssim

def reconstruction_loss(image, image_pred, invalid_mask, ssim_factor=0.85):
    """Compute image reconstruction loss.
    """
    invalid_maskc = invalid_mask.expand(-1, image.shape[1], -1, -1)
    l1_loss = torch.nn.functional.l1_loss(image_pred[~invalid_maskc], image[~invalid_maskc])

    # NOTE: SSIM requires neighbor pixels that might be invalid. Thus we need to
    # dilate the invalid mask to rule out these pixels.
    ssim_patch_size = 3
    dilated_mask = torch.nn.functional.avg_pool2d(
        invalid_mask.float(), ssim_patch_size, stride=1, padding=ssim_patch_size//2)
    dilated_mask = dilated_mask > 0
    dilated_maskc = dilated_mask.expand(-1, image.shape[1], -1, -1)

    ssim_image = SSIM(image_pred, image, ssim_patch_size)
    ssim_loss = torch.mean(ssim_image[~dilated_maskc])

    loss = ssim_factor * ssim_loss + (1.0 - ssim_factor) * l1_loss

    return loss

def smoothness_loss(image, output, alpha):
    """Compute edge-aware smoothness loss.

    This is a form of anistropic total variation where the norm of a gradient
    element is set to the L1 norm (vanilla TV is usually L2). See here:
    https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    assert(len(image.shape) == 4) # (batch, channel, rows, cols)
    assert(len(output.shape) == 4) # (batch, channel, rows, cols)

    output_channels = output.shape[1]

    blur = image_utils.GaussianBlur(image.device, 5, 1.0, 3)
    image_smooth = blur(image)

    image_gx = image_utils.forward_gradx(image_smooth)
    image_gy = image_utils.forward_grady(image_smooth)

    output_gx = image_utils.forward_gradx(output)
    output_gy = image_utils.forward_grady(output)

    channel_dim = 1
    weights_x = torch.exp(-alpha * torch.mean(torch.abs(image_gx), channel_dim, keepdim=True))
    weights_y = torch.exp(-alpha * torch.mean(torch.abs(image_gy), channel_dim, keepdim=True))

    smoothness_x = torch.mean(torch.abs(output_gx) * weights_x.expand(-1, output_channels, -1, -1))
    smoothness_y = torch.mean(torch.abs(output_gy) * weights_y.expand(-1, output_channels, -1, -1))

    return smoothness_x + smoothness_y
