# Copyright 2021 Massachusetts Institute of Technology
#
# @file losses.py
# @author W. Nicholas Greene
# @date 2020-02-19 13:53:50 (Wed)

import torch
import torch.nn as tnn

from stereo import image_predictor as ip

from utils import losses

def supervised_idepthmap_loss(idepthmap, truth, truth_mask, scale_factor=1000, normalize=True):
    """Compute supervised idepthmap loss.

    Will resize disparity estimate to fit the ground truth.
    """
    assert(torch.sum(truth_mask) > 0)

    batch_size = truth.shape[0]
    channels = truth.shape[1]
    truth_size = truth.shape[-2:]

    mean_idepths = torch.ones(truth.shape, device=truth.device)
    if normalize:
        # Compute mean using only truth_mask values.
        mean_idepths = (truth * truth_mask).sum(
            dim=[1, 2, 3], keepdim=True) / truth_mask.sum(dim=[1, 2, 3], keepdim=True)
        mean_idepths = mean_idepths.repeat(1, channels, truth_size[0], truth_size[1])

    idepthmap_resized = torch.nn.functional.interpolate(
        idepthmap, size=truth_size,
        mode="bilinear", align_corners=False)
    supervised_loss_lvl = losses.pseudo_huber_loss(
        scale_factor * truth[truth_mask] / mean_idepths[truth_mask],
        scale_factor * idepthmap_resized[truth_mask] / mean_idepths[truth_mask])
    assert(torch.isnan(supervised_loss_lvl) == False)

    return supervised_loss_lvl

def get_occlusion_mask(K, T_right_in_left,
                       left_idepthmap, left_invalid_mask,
                       right_idepthmap, right_invalid_mask):
    """Get left mask that is 1 where a pixel is occluded in the right view.

    See Depth from Videos in the Wild (Gordon et al. 2019).
    """
    batch = left_idepthmap.shape[0]
    rows = left_idepthmap.shape[-2]
    cols = left_idepthmap.shape[-1]

    # Project left idepthmap into right frame. For each left pixel (u, v, id),
    # we get (u', v', id') where (u', v') is the projection of (u, v) into the
    # right frame and id' is the idepth of the point in the right frame.
    idepthmap_projector = ip.IDepthmapProjector()
    uv_prime, id_prime, prime_invalid_mask = idepthmap_projector(
        K, T_right_in_left, left_idepthmap)

    # We now sample right_idepthmap at (u', v') to get id_pred.
    id_pred = torch.nn.functional.grid_sample(
        right_idepthmap, uv_prime, mode="bilinear",
        padding_mode="border", align_corners=False)

    # Occluded pixels occur when the difference between id_prime < id_pred. We
    # want some buffer though, so compute the average idepth difference and mark
    # pixels occluded if their idepth diff is greater.
    id_diff = id_pred - id_prime # Positive diff marks occlusion.
    occlusion_threshold = torch.mean(torch.abs(id_diff.view(batch, -1)), dim=1)
    occlusion_threshold = occlusion_threshold.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    occlusion_threshold = occlusion_threshold.expand(-1, 1, rows, cols)
    left_mask = id_diff > occlusion_threshold # Positive diff greater than thresh marks occlusion.

    # Combine with other masks.
    # left_mask = left_mask | left_invalid_mask | right_invalid_mask | prime_invalid_mask
    left_mask = left_mask | prime_invalid_mask

    return left_mask


def reconstruction_loss(T_right_in_left, K, left_image, right_image, left_idepthmap,
                        left_occlusion_mask):
    """Compute the reconstruction loss by predicting the left image given the right
    image and the left idepthmap.

    All idepthmaps will be resized to the corresponding image size.
    """
    # Resize idepthmap map.
    left_idepthmap_resized = torch.nn.functional.interpolate(
        left_idepthmap, size=left_image.shape[-2:],
        mode="bilinear", align_corners=False)

    # Resize occlusion mask.
    left_occlusion_mask_resizedf = torch.nn.functional.interpolate(
        left_occlusion_mask.float(), size=left_image.shape[-2:],
        mode="bilinear", align_corners=False)
    left_occlusion_mask_resized = left_occlusion_mask_resizedf > 0.5

    image_predictor = ip.IDepthImagePredictor()
    left_image_pred, left_image_pred_mask = image_predictor(
        K, T_right_in_left, left_idepthmap_resized, right_image)

    loss = losses.reconstruction_loss(
        left_image, left_image_pred, left_occlusion_mask_resized)

    return loss, left_image_pred

def left_right_idepthmap_consistency_losses(
        T_right_in_left, T_left_in_right, K_pyr,
        left_idepthmap_pyr, left_occlusion_mask_pyr,
        right_idepthmap_pyr, right_occlusion_mask_pyr):
    """Compute left/right geometric consistency loss between idepthmaps.
    """
    num_levels = len(left_idepthmap_pyr)

    loss = 0.0
    idepthmap_projector = ip.IDepthmapProjector()
    for lvl in range(num_levels):
        if left_idepthmap_pyr[lvl] is None:
            continue

        # Project left idepthmap into right frame.
        left_to_right_pixels, left_to_right_idepths, _ = idepthmap_projector(
            K_pyr[lvl], T_right_in_left, left_idepthmap_pyr[lvl])

        # Use pixel coordinates to sample right_idepthmap and right_occlusion_mask.
        right_sampled_idepths = torch.nn.functional.grid_sample(
            right_idepthmap_pyr[lvl], left_to_right_pixels, mode="bilinear",
            padding_mode="border", align_corners=False)
        right_sampled_occlusion_mask = torch.nn.functional.grid_sample(
            right_occlusion_mask_pyr[lvl].float(), left_to_right_pixels, mode="bilinear",
            padding_mode="border", align_corners=False) > 0

        # Compute loss. Loss is only valid if pixel is unoccluded in both frames.
        right_unocclusion_mask = ~left_occlusion_mask_pyr[lvl] & ~right_sampled_occlusion_mask
        right_loss = torch.nn.functional.l1_loss(
            left_to_right_idepths[right_unocclusion_mask],
            right_sampled_idepths[right_unocclusion_mask])

        # Project right idepthmap into left frame.
        right_to_left_pixels, right_to_left_idepths, _ = idepthmap_projector(
            K_pyr[lvl], T_left_in_right, right_idepthmap_pyr[lvl])

        # Use pixel coordinates to sample left_idepthmap and left_occlusion_mask.
        left_sampled_idepths = torch.nn.functional.grid_sample(
            left_idepthmap_pyr[lvl], right_to_left_pixels, mode="bilinear",
            padding_mode="border", align_corners=False)
        left_sampled_occlusion_mask = torch.nn.functional.grid_sample(
            left_occlusion_mask_pyr[lvl].float(), right_to_left_pixels, mode="bilinear",
            padding_mode="border", align_corners=False) > 0

        # Compute loss. Loss is only valid if pixel is unoccluded in both frames.
        left_unocclusion_mask = ~right_occlusion_mask_pyr[lvl] & ~left_sampled_occlusion_mask
        left_loss = torch.nn.functional.l1_loss(
            right_to_left_idepths[left_unocclusion_mask],
            left_sampled_idepths[left_unocclusion_mask])

        loss += right_loss + left_loss

    return loss
