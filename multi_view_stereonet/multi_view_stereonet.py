# Copyright 2021 Massachusetts Institute of Technology
#
# @file mvstereonet.py
# @author W. Nicholas Greene
# @date 2020-02-19 13:53:38 (Wed)

from typing import Dict, List, Optional

import numpy as np

import torch
import torch.nn as tnn
from torch.jit import Final

import torchvision as tv

from stereo import image_predictor as ip

from utils import resnet

# Type alias for a list of tensors. Type hints are needed to TorchScript-ify
# some methods.
ListTensor = List[torch.Tensor]

def feature_normalizer2d(out_channels):
    # return tnn.BatchNorm2d(out_channels, track_running_stats=True)
    return tnn.GroupNorm(out_channels // 8, out_channels)

def feature_normalizer3d(out_channels):
    # return tnn.modules.batchnorm.BatchNorm3d(out_channels, track_running_stats=True)
    return tnn.GroupNorm(out_channels // 8, out_channels)

def conv2d_padded(in_channels, out_channels, kernel_size, stride=1, bias=True, gain=1.0):
    """Returns a Conv2d layer that does not change the spatial input dimensions.

    The padding is set based on the kernel size such that the spatial input
    dimensions are unchanged.
    """
    conv = tnn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, 1, 1, bias)
    conv.weight.data.normal_(0, 0.01)

    # Special cases?
    if conv.weight.data.shape == torch.Size([1, 5, 1, 1]):
        tnn.init.constant_(conv.weight, 0.2)

    if conv.bias is not None:
        tnn.init.zeros_(conv.bias)
    return conv

def resnet_block(in_channels, out_channels, downsample=False, dilation=1, gain=1.0, bias=True):
    """Returns a resnet block.
    """
    stride = 1
    downsampler = None
    if downsample:
        stride = 2
        downsampler = resnet.conv1x1(in_channels, out_channels, stride=stride)
    elif (in_channels != out_channels):
        downsampler = resnet.conv1x1(in_channels, out_channels, stride=stride)

    block = resnet.SimpleBasicBlock(in_channels, out_channels, stride, downsampler,
                                    groups=1, base_width=64, dilation=dilation,
                                    norm_layer=None, bias=bias)
    block.relu = tnn.LeakyReLU(0.2, inplace=True)

    block.bn1 = feature_normalizer2d(out_channels)

    block.conv1.weight.data.normal_(0, 0.01)

    if block.conv1.bias is not None:
        tnn.init.zeros_(block.conv1.bias)

    # if downsample:
    #     tnn.init.xavier_uniform_(block.downsample.weight, gain=gain)

    return block

class FeatureNetwork(tnn.Module):
    """Extracts high-level features from input images to perform stereo with.

    Based on StereoNet (Khamis 2018).
    """
    def __init__(self, in_channels):
        super(FeatureNetwork, self).__init__()

        self.in_channels = in_channels
        self.channels = [self.in_channels, 32, 32, 32, 32]

        # Bunch of downsampling.
        self.pyramid_level = 4
        self.conv0 = conv2d_padded(self.channels[0], self.channels[1], 5, 2, False)
        self.conv1 = conv2d_padded(self.channels[1], self.channels[2], 5, 2, False)
        self.conv2 = conv2d_padded(self.channels[2], self.channels[3], 5, 2, False)
        self.conv3 = conv2d_padded(self.channels[3], self.channels[4], 5, 2, False)

        # Bunch of resnet blocks.
        self.res0 = resnet_block(self.channels[-1], self.channels[-1], False, 1, 1.0, False)
        self.res1 = resnet_block(self.channels[-1], self.channels[-1], False, 1, 1.0, False)
        self.res2 = resnet_block(self.channels[-1], self.channels[-1], False, 1, 1.0, False)
        self.res3 = resnet_block(self.channels[-1], self.channels[-1], False, 1, 1.0, False)
        self.res4 = resnet_block(self.channels[-1], self.channels[-1], False, 1, 1.0, False)
        self.res5 = resnet_block(self.channels[-1], self.channels[-1], False, 1, 1.0, False)

        # conv_final doesn't have batch_norm or activation.
        self.conv_final = conv2d_padded(self.channels[-1], self.channels[-1], 3)

        return

    def forward(self, x):
        pyramid = [x]

        # Downsampling layers.
        pyramid.append(self.conv0(pyramid[-1]))
        pyramid.append(self.conv1(pyramid[-1]))
        pyramid.append(self.conv2(pyramid[-1]))
        feats = self.conv3(pyramid[-1])

        # Resnet layers.
        feats = self.res0(feats)
        feats = self.res1(feats)
        feats = self.res2(feats)
        feats = self.res3(feats)
        feats = self.res4(feats)
        feats = self.res5(feats)

        # Final conv layer.
        pyramid.append(self.conv_final(feats))

        return pyramid

def create_idepth_samples(T_right_in_left: torch.Tensor, K: torch.Tensor,
                          rows: int, cols: int, num_idepth_samples: int):
    """Create idepth samples per batch element given camera intrinsics/extrinsics.
    """
    batch_size = T_right_in_left.shape[0]

    # Compute maximum idepth by assuming maximum disparity of
    # num_idepth_samples-1.
    max_disparity = (num_idepth_samples - 1) * torch.ones((batch_size, 1, rows, cols), device=K.device)
    max_idepthmap = ip.disparity_to_idepth(K, T_right_in_left, max_disparity)
    max_idepthmap = (max_idepthmap > 0).float() * max_idepthmap

    # Compute mean over valid idepths.
    max_idepthmap = max_idepthmap.reshape(batch_size, -1)
    sum_idepthmap = torch.sum(max_idepthmap, 1)
    count_idepthmap = torch.sum(max_idepthmap > 0, 1)
    mean_idepths = sum_idepthmap / count_idepthmap
    max_idepths = mean_idepths
    max_idepths[max_idepths > 2.0] = torch.tensor(2.0, dtype=torch.float)

    # Make sure samples don't go behind right camera.
    tz_right_in_left = T_right_in_left[:, 2, 3]
    tz_mask = 1.0 / max_idepths < tz_right_in_left
    max_idepths[tz_mask] = 1.0 / tz_right_in_left[tz_mask]

    min_idepths = torch.zeros(batch_size, device=K.device)

    idepth_deltas = (max_idepths - min_idepths) / (num_idepth_samples - 1) # (batch)
    idepth_deltas = idepth_deltas.unsqueeze(1).repeat(1, num_idepth_samples) # (batch, idepths)

    idepth_samples = torch.arange(0.0, num_idepth_samples).to(K.device) # (idepths)
    idepth_samples = idepth_samples.unsqueeze(0).repeat(batch_size, 1) # (batch, idepths)
    idepth_samples = idepth_samples * idepth_deltas + min_idepths.unsqueeze(1).repeat(1, num_idepth_samples) # (batch, idepths)

    return idepth_samples

def create_plane_sweep_homographies(
        T_right_in_left: torch.Tensor, K: torch.Tensor, idepth_samples: torch.Tensor,
        image_size: List[int]):
    """Create family of homographies from idepth samples to be used in plane sweep.
    """
    batch_size = T_right_in_left.shape[0]
    rows = image_size[0]
    cols = image_size[1]
    num_idepth_samples = idepth_samples.shape[1]

    T_left_in_right = torch.inverse(T_right_in_left)

    # Compute feature volume.
    # NOTE: In order to parallelize across multiple candidate idepths, we
    # will combine the idepth dimension with the batch dimension.
    K_rebatched = K.repeat(num_idepth_samples, 1, 1)
    T_left_in_right_rebatched = T_left_in_right.repeat(num_idepth_samples, 1, 1)
    idepth_samples_rebatched = idepth_samples.permute(1, 0).reshape(-1) # (idepths*batch)

    # Compute homographies. (idepths * batch, 3, 3)
    H_left_in_right_rebatched = ip.get_fronto_parallel_homography(
        K_rebatched[:, :3, :3], K_rebatched[:, :3, :3],
        T_left_in_right_rebatched, idepth_samples_rebatched)

    H_left_in_right = H_left_in_right_rebatched.reshape(num_idepth_samples, batch_size, 3, 3)
    H_left_in_right = H_left_in_right.permute(1, 0, 2, 3) # (batch_size, idepths, 3, 3)

    return H_left_in_right

class PlaneSweepWarper(tnn.Module):
    """Converts an image into a warped volume using a family of homographies to be
    used in Plane Sweep.
    """
    def __init__(self):
        super(PlaneSweepWarper, self).__init__()
        self.image_predictor = ip.HomographyImagePredictor()
        return

    def forward(self, right_image: torch.Tensor, H_left_in_right: torch.Tensor):
        batch_size = right_image.shape[0]
        channels = right_image.shape[1]
        rows = right_image.shape[-2]
        cols = right_image.shape[-1]
        num_idepth_samples = H_left_in_right.shape[1]

        # NOTE: In order to parallelize across multiple candidate idepths, we
        # will combine the idepth dimension with the batch dimension.
        right_image_rebatched = right_image.repeat(num_idepth_samples, 1, 1, 1)
        H_left_in_right_rebatched = H_left_in_right.permute(1, 0, 2, 3).reshape(-1, 3, 3) # (idepths * batch, 3, 3)

        # right_warped_rebatched should be (idepths * batch, channels, rows, cols).
        right_warped_rebatched, right_warped_mask_rebatched = self.image_predictor(
            H_left_in_right_rebatched, right_image_rebatched)

        right_warped_rebatched = right_warped_rebatched.reshape(
            num_idepth_samples, batch_size, channels, rows, cols)
        right_warped_mask_rebatched = right_warped_mask_rebatched.reshape(
            num_idepth_samples, batch_size, 1, rows, cols)

        # Volume should (batch, channels, idepths, rows, cols).
        right_warped_volume = right_warped_rebatched.permute(1, 2, 0, 3, 4)
        right_warped_mask = right_warped_mask_rebatched.permute(1, 2, 0, 3, 4)

        # Make invalid voxels 0.0.
        # Tensor wrapper around 0.0 needed for TorchScript.
        right_warped_mask_with_channels = right_warped_mask.repeat(1, channels, 1, 1, 1)
        right_warped_volume = (~right_warped_mask_with_channels).float() * right_warped_volume

        return right_warped_volume, right_warped_mask

class IncrementalFastGeometryAwareFeatureNetwork(tnn.Module):
    """Extracts high-level features from a right image plus epipolar geometry.
    """
    def __init__(self, feature_extractor):
        super(IncrementalFastGeometryAwareFeatureNetwork, self).__init__()
        self.warper = PlaneSweepWarper()
        self.feature_extractor = feature_extractor
        self.refiner = FeatureRefiner(32)
        return

    def forward(self, T_right_in_left: torch.Tensor, K_pyr: ListTensor,
                right_image_pyr: ListTensor, idepth_samples: torch.Tensor):
        batch_size = T_right_in_left.shape[0]
        num_idepth_samples = idepth_samples.shape[1]

        # Warp full-res image using min_idepth.
        # right_image0_idepth0 is (batch, channels, 1, rows, cols).
        H_min_idepth0 = create_plane_sweep_homographies(
            T_right_in_left, K_pyr[0], idepth_samples[:, 0].unsqueeze(1), right_image_pyr[0].shape[-2:])

        right_image0_idepth0, right_mask0 = self.warper(right_image_pyr[0], H_min_idepth0)
        right_image0_idepth0 = right_image0_idepth0.squeeze(2) # (batch, channels, rows, cols).

        # Extract features from warped image.
        right_features_idepth0 = self.feature_extractor(right_image0_idepth0)[-1]

        # Resize mask.
        right_downsampled_mask0f = tnn.functional.interpolate(
            right_mask0.squeeze(1).float(), size=right_features_idepth0.shape[-2:],
            mode="bilinear", align_corners=False)
        right_downsampled_mask0 = right_downsampled_mask0f > 0.5 # (batch, 1, rows, cols)

        # Compute homographies.
        H_left_in_right = create_plane_sweep_homographies(
            T_right_in_left, K_pyr[-1], idepth_samples, right_features_idepth0.shape[-2:])

        # Convert image to warped volume.
        # Volume is (batch, channels, idepths, rows, cols).
        right_image_volume, right_mask_volume = self.warper(right_image_pyr[-1], H_left_in_right)

        # Extract features for other idepth samples incrementally.
        right_features = [right_features_idepth0.unsqueeze(2)]
        for idepth_idx in range(1, num_idepth_samples):
            H_idx = H_left_in_right[:, idepth_idx, :, :].unsqueeze(1)
            H_inv = torch.inverse(H_left_in_right[:, idepth_idx - 1, :, :].unsqueeze(1))
            H_inc = torch.matmul(H_inv, H_idx)

            # Incrementally warp features.
            right_features_idx, right_mask_idx = self.warper(right_features[-1].squeeze(2), H_inc)

            # Refine features.
            right_features_idx = self.refiner(right_image_volume[:, :, idepth_idx, :, :], right_features_idx.squeeze(2))

            right_features.append(right_features_idx.unsqueeze(2))

        # Form feature volume by concatenating along idepth dimension.
        right_feature_volume = torch.cat(right_features, dim=2) # (batch, channels, idepths, rows, cols)

        # Make invalid voxels 0.0.
        channels = right_feature_volume.shape[1]
        right_mask_with_channels = right_mask_volume.repeat(1, channels, 1, 1, 1)
        right_feature_volume = (~right_mask_with_channels).float() * right_feature_volume

        return right_feature_volume, right_mask_volume.squeeze(1)

class CostVolumeFilter(tnn.Module):
    """Filters stereo cost volume using 3d convolutions.
    """
    def conv3d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        conv = tnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=kernel_size//2, dilation=1, groups=1, bias=bias)
        conv.weight.data.normal_(0, 0.01)

        # Special cases?
        if conv.weight.data.shape == torch.Size([1, 5, 1, 1]):
            tnn.init.constant_(conv.weight, 0.2)

        if conv.bias is not None:
            tnn.init.zeros_(conv.bias)
        return conv

    def __init__(self, channels=1):
        super(CostVolumeFilter, self).__init__()

        self.channels = channels

        self.relu = tnn.LeakyReLU(0.2, inplace=True)

        self.conv0 = self.conv3d(self.channels, self.channels)
        self.bn0 = feature_normalizer3d(self.channels)

        self.conv1 = self.conv3d(self.channels, self.channels)
        self.bn1 = feature_normalizer3d(self.channels)

        self.conv2 = self.conv3d(self.channels, self.channels)
        self.bn2 = feature_normalizer3d(self.channels)

        self.conv3 = self.conv3d(self.channels, self.channels)
        self.bn3 = feature_normalizer3d(self.channels)

        self.conv4 = self.conv3d(self.channels, 1)

        return

    def forward(self, volume):
        # Assumes volume = [batch, channels, depth, rows, cols].
        assert(len(volume.shape) == 5)

        # volume = volume.norm(dim=1, keepdim=True)

        volume = self.relu(self.bn0(self.conv0(volume)))
        volume = self.relu(self.bn1(self.conv1(volume)))
        volume = self.relu(self.bn2(self.conv2(volume)))
        volume = self.relu(self.bn3(self.conv3(volume)))
        volume = self.conv4(volume)

        return torch.squeeze(volume, 1) # Get rid of channel dimension.

class Upsampler(tnn.Module):
    """Upsamples a tensor using bilinear interpolation followed by a conv2d + relu.
    """
    def __init__(self, channels, refine=True, relu=True):
        super(Upsampler, self).__init__()
        self.channels = channels
        self.kernel_size = 3

        self.refiner = tnn.Identity()
        if refine:
            self.refiner = conv2d_padded(channels, channels, self.kernel_size, 1, True)
            tnn.init.dirac_(self.refiner.weight)

        self.relu = relu

        return

    def forward(self, x: torch.Tensor, output_size: List[int]):
        x = tnn.functional.interpolate(
            x, size=output_size, mode="bilinear", align_corners=False)

        x = self.refiner(x)
        if self.relu:
            x = tnn.functional.relu(x)

        return x

class MaskUpsampler(tnn.Module):
    """Upsamples a boolean mask.
    """
    def __init__(self):
        super(MaskUpsampler, self).__init__()
        return

    def forward(self, mask: torch.Tensor, output_size: List[int]):
        maskf = mask.float()
        upsampled_maskf = tnn.functional.interpolate(
            maskf, size=output_size, mode="bilinear", align_corners=False)

        upsampled_mask = upsampled_maskf > 0.5

        return upsampled_mask

class FeatureRefiner(tnn.Module):
    """Refines features using image as guidance.
    """
    def __init__(self, feature_channels, gain=1.0, kernel_size=3):
        super(FeatureRefiner, self).__init__()

        self.feature_channels = feature_channels

        self.kernel_size = kernel_size

        # Input will be color + idepthmap.
        self.conv0 = conv2d_padded(self.feature_channels + 3, 32, self.kernel_size, 1, True, gain)
        self.bn0 = feature_normalizer2d(32)
        self.relu = tnn.LeakyReLU(0.2, inplace=True)

        self.dilations = [1, 2, 4, 8, 1, 1]
        self.res0 = resnet_block(32, 32, False, self.dilations[0], gain)
        # self.res1 = resnet_block(32, 32, False, self.dilations[1], gain)
        # self.res2 = resnet_block(32, 32, False, self.dilations[2], gain)
        # self.res3 = resnet_block(32, 32, False, self.dilations[3], gain)
        # self.res4 = resnet_block(32, 32, False, self.dilations[4], gain)
        # self.res5 = resnet_block(32, 32, False, self.dilations[5], gain)
        self.conv_final = conv2d_padded(32, self.feature_channels, self.kernel_size, 1, True, gain)

        return

    def forward(self, image, features):
        x = torch.cat([image, features], dim=1)

        x = self.relu(self.bn0(self.conv0(x)))

        x = self.res0(x)
        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.res3(x)
        # x = self.res4(x)
        # x = self.res5(x)

        delta = self.conv_final(x)

        new_features = features + delta

        return new_features

class IDepthmapRefiner(tnn.Module):
    """Refines idepthmaps using image/features as guidance.
    """
    def __init__(self, image_channels, gain, kernel_size=3):
        super(IDepthmapRefiner, self).__init__()

        self.image_channels = image_channels

        self.kernel_size = kernel_size

        # Input will be color + idepthmap.
        self.conv0 = conv2d_padded(self.image_channels + 1, 32, self.kernel_size, 1, True, gain)
        self.bn0 = feature_normalizer2d(32)
        self.relu = tnn.LeakyReLU(0.2, inplace=True)

        self.dilations = [1, 2, 4, 8, 1, 1]
        self.res0 = resnet_block(32, 32, False, self.dilations[0], gain)
        self.res1 = resnet_block(32, 32, False, self.dilations[1], gain)
        self.res2 = resnet_block(32, 32, False, self.dilations[2], gain)
        self.res3 = resnet_block(32, 32, False, self.dilations[3], gain)
        self.res4 = resnet_block(32, 32, False, self.dilations[4], gain)
        self.res5 = resnet_block(32, 32, False, self.dilations[5], gain)
        self.conv_final = conv2d_padded(32, 1, self.kernel_size, 1, True, gain)

        return

    def forward(self, image, idepthmap):
        x = torch.cat([image, idepthmap], dim=1)

        x = self.relu(self.bn0(self.conv0(x)))

        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        delta = self.conv_final(x)

        new_idepthmap = tnn.functional.relu(idepthmap + delta)

        return new_idepthmap

def extract_idepthmap(cost_volume, idepthmap_volume):
    """ Extract idepths from cost volume using softmin. """
    softmin_beta = 1e0
    probs = tnn.functional.softmin(softmin_beta * cost_volume, dim=1)
    idepths = probs * idepthmap_volume
    left_idepthmap = torch.sum(idepths, dim=1, keepdim=True)
    return left_idepthmap

class MultiViewStereoNet(tnn.Module):
    """Layer that performs multi-view stereo matching given a set of features.
    """
    num_levels: Final[int]

    def __init__(self):
        super(MultiViewStereoNet, self).__init__()

        self.min_idepth = 0.0

        self.num_levels = 5

        self.left_feature_extractor = FeatureNetwork(3)
        self.right_feature_extractor = IncrementalFastGeometryAwareFeatureNetwork(self.left_feature_extractor)
        assert(len(self.left_feature_extractor.channels) == self.num_levels)

        # Level 4.
        self.volume_filter4 = CostVolumeFilter(32)
        self.refiner4 = IDepthmapRefiner(self.left_feature_extractor.channels[-1] + 3, 1.0)

        # Level 3.
        self.idepthmap_upsampler3 = Upsampler(1, False, False)
        self.mask_upsampler3 = MaskUpsampler()
        self.refiner3 = IDepthmapRefiner(self.left_feature_extractor.channels[3] + 3, 1.0)

        # Level 2.
        self.idepthmap_upsampler2 = Upsampler(1, False, False)
        self.mask_upsampler2 = MaskUpsampler()
        self.refiner2 = IDepthmapRefiner(self.left_feature_extractor.channels[2] + 3, 1.0)

        # Level 1.
        self.idepthmap_upsampler1 = Upsampler(1, False, False)
        self.mask_upsampler1 = MaskUpsampler()
        self.refiner1 = IDepthmapRefiner(self.left_feature_extractor.channels[1] + 3, 1.0)

        # Level 0.
        self.idepthmap_upsampler0 = Upsampler(1, False, False)
        self.mask_upsampler0 = MaskUpsampler()
        self.refiner0 = IDepthmapRefiner(3, 1.0)

        self.debug_warper = PlaneSweepWarper()

        return

    def forward(self,
                left_image_pyr: ListTensor,
                K_pyr: ListTensor,
                T_right_in_lefts: ListTensor,
                right_image_pyrs: List[ListTensor],
                num_idepth_samples: int,
                do_cost_volume_filter: bool,
                do_refiners: List[bool]) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Returns estimated left idepth map.
        """
        assert(len(K_pyr) == self.num_levels)
        assert(len(left_image_pyr) == self.num_levels)

        # Extract left features. Final volumes should be (batch, channels, idepths, rows, cols)
        left_feature_pyr = self.left_feature_extractor(left_image_pyr[0]) # (batch, channels, rows, cols)
        left_feature_volume = left_feature_pyr[-1].unsqueeze(2).repeat(1, 1, num_idepth_samples, 1, 1)

        batch = left_feature_pyr[-1].shape[0]
        rows4 = left_feature_pyr[-1].shape[-2]
        cols4 = left_feature_pyr[-1].shape[-1]

        # Level 4.
        # Compute idepthmaps for each right camera and then average.
        idepthmap4_raw_sum = torch.zeros((batch, 1, rows4, cols4), device=left_image_pyr[-1].device)
        idepthmap4_sum = torch.zeros((batch, 1, rows4, cols4), device=left_image_pyr[-1].device)
        mask4_sum = torch.zeros((batch, num_idepth_samples, rows4, cols4), device=left_image_pyr[-1].device)
        for right_idx in range(len(T_right_in_lefts)):
            # Normalize by by baseline.
            T_right_in_left_idx = torch.clone(T_right_in_lefts[right_idx])

            baseline = torch.sqrt(torch.sum(T_right_in_left_idx[:, :3, 3]**2, 1))
            baseline3 = torch.unsqueeze(baseline, dim=1)
            baseline3 = baseline3.repeat(1, 3)
            T_right_in_left_idx[:, 0:3, 3] /= baseline3

            # Create idepth samples
            idepth_samples = create_idepth_samples(
                T_right_in_left_idx, K_pyr[-1], left_image_pyr[-1].shape[-2], left_image_pyr[-1].shape[-1],
                num_idepth_samples)
            idepthmap_volume = idepth_samples.unsqueeze(2).unsqueeze(3).repeat(
                1, 1, left_feature_pyr[-1].shape[-2], left_feature_pyr[-1].shape[-1]) # (batch, idepth, rows, cols)

            # Extract right features. Final volumes should be (batch, channels, idepths, rows, cols)
            # right_feature_volume, right_feature_volume_mask = self.right_feature_extractor(
            #     T_right_in_left_idx, K_pyr[0], right_image_pyrs[right_idx][0], idepth_samples)
            right_feature_volume, right_feature_volume_mask = self.right_feature_extractor(
                T_right_in_left_idx, K_pyr, right_image_pyrs[right_idx], idepth_samples)

            # Compute cost volume.
            cost_volume_idx = torch.abs(left_feature_volume - right_feature_volume)

            # Zero-out invalid voxels.
            channels = right_feature_volume.shape[1]
            right_mask_with_channels = right_feature_volume_mask.unsqueeze(1).repeat(1, channels, 1, 1, 1)
            cost_volume_idx = (~right_mask_with_channels) * cost_volume_idx

            # Filter cost volume.
            if do_cost_volume_filter:
                cost_volume = self.volume_filter4(cost_volume_idx)
            else:
                cost_volume = torch.norm(cost_volume_idx, dim=1)
            assert(len(cost_volume.shape) == 4)
            assert(cost_volume.shape[1] == num_idepth_samples)

            left_idepthmap4_raw = extract_idepthmap(cost_volume, idepthmap_volume)
            left_idepthmap4_mask = right_feature_volume_mask

            if do_refiners[4]:
                # Refine. Scale up idepths before refining to not mess around in gains.
                idepth_scale_factor4 = K_pyr[-1][:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                idepth_scale_factor4 = idepth_scale_factor4.repeat(1, 1, left_idepthmap4_raw.shape[-2], left_idepthmap4_raw.shape[-1])
                left_idepthmap4_scaled = self.refiner4(
                    torch.cat((left_image_pyr[-1], left_feature_pyr[-1]), dim=1), left_idepthmap4_raw * idepth_scale_factor4)
                left_idepthmap4 = left_idepthmap4_scaled / idepth_scale_factor4
            else:
                left_idepthmap4 = left_idepthmap4_raw

            # Scale back by baseline.
            baselinehw = baseline.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            baselinehw = baselinehw.repeat(1, 1, rows4, cols4)
            left_idepthmap4_raw /= baselinehw
            left_idepthmap4 /= baselinehw

            idepthmap4_raw_sum += left_idepthmap4_raw
            idepthmap4_sum += left_idepthmap4
            mask4_sum += left_idepthmap4_mask.float()

        left_idepthmap4_raw = idepthmap4_raw_sum / len(T_right_in_lefts)
        left_idepthmap4 = idepthmap4_sum / len(T_right_in_lefts)
        left_idepthmap4_mask = (mask4_sum / len(T_right_in_lefts)) > 0.5

        # Level 3.
        left_idepthmap3_prior = self.idepthmap_upsampler3(left_idepthmap4, left_image_pyr[3].shape[-2:])
        left_idepthmap3_mask = self.mask_upsampler3(left_idepthmap4_mask, left_image_pyr[3].shape[-2:])

        if do_refiners[3]:
            # Refine. Scale up idepths before refining to not mess around in gains.
            idepth_scale_factor3 = K_pyr[3][:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            idepth_scale_factor3 = idepth_scale_factor3.repeat(1, 1, left_idepthmap3_prior.shape[-2], left_idepthmap3_prior.shape[-1])
            left_idepthmap3_scaled = self.refiner3(
                torch.cat((left_image_pyr[3], left_feature_pyr[3]), dim=1), left_idepthmap3_prior * idepth_scale_factor3)
            left_idepthmap3 = left_idepthmap3_scaled / idepth_scale_factor3
        else:
            left_idepthmap3 = left_idepthmap3_prior

        # Level 2.
        left_idepthmap2_prior = self.idepthmap_upsampler2(left_idepthmap3, left_image_pyr[2].shape[-2:])
        left_idepthmap2_mask = self.mask_upsampler2(left_idepthmap3_mask, left_image_pyr[2].shape[-2:])

        if do_refiners[2]:
            # Refine. Scale up idepths before refining to not mess around in gains.
            idepth_scale_factor2 = K_pyr[2][:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            idepth_scale_factor2 = idepth_scale_factor2.repeat(1, 1, left_idepthmap2_prior.shape[-2], left_idepthmap2_prior.shape[-1])
            left_idepthmap2_scaled = self.refiner2(
                torch.cat((left_image_pyr[2], left_feature_pyr[2]), dim=1), left_idepthmap2_prior * idepth_scale_factor2)
            left_idepthmap2 = left_idepthmap2_scaled / idepth_scale_factor2
        else:
            left_idepthmap2 = left_idepthmap2_prior

        # Level 1.
        left_idepthmap1_prior = self.idepthmap_upsampler1(left_idepthmap2, left_image_pyr[1].shape[-2:])
        left_idepthmap1_mask = self.mask_upsampler1(left_idepthmap2_mask, left_image_pyr[1].shape[-2:])

        if do_refiners[1]:
            # Refine. Scale up idepths before refining to not mess around in gains.
            idepth_scale_factor1 = K_pyr[1][:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            idepth_scale_factor1 = idepth_scale_factor1.repeat(1, 1, left_idepthmap1_prior.shape[-2], left_idepthmap1_prior.shape[-1])
            left_idepthmap1_scaled = self.refiner1(
                torch.cat((left_image_pyr[1], left_feature_pyr[1]), dim=1), left_idepthmap1_prior * idepth_scale_factor1)
            left_idepthmap1 = left_idepthmap1_scaled / idepth_scale_factor1
        else:
            left_idepthmap1 = left_idepthmap1_prior

        # Level 0.
        left_idepthmap0_prior = self.idepthmap_upsampler0(left_idepthmap1, left_image_pyr[0].shape[-2:])
        left_idepthmap0_mask = self.mask_upsampler0(left_idepthmap1_mask, left_image_pyr[0].shape[-2:])

        if do_refiners[0]:
            # Refine. Scale up idepths before refining to not mess around in gains.
            idepth_scale_factor0 = K_pyr[0][:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            idepth_scale_factor0 = idepth_scale_factor0.repeat(1, 1, left_idepthmap0_prior.shape[-2], left_idepthmap0_prior.shape[-1])
            left_idepthmap0_scaled = self.refiner0(left_image_pyr[0], left_idepthmap0_prior * idepth_scale_factor0)
            left_idepthmap0 = left_idepthmap0_scaled / idepth_scale_factor0
        else:
            left_idepthmap0 = left_idepthmap0_prior

        # TorchScript may infer type of output incorrectly, so use type hints to
        # be explicit.
        outputs_left_idepthmap_pyr: List[Optional[torch.Tensor]] = [left_idepthmap0, left_idepthmap1, left_idepthmap2, left_idepthmap3, left_idepthmap4]
        outputs_left_idepthmap_raw_pyr: List[Optional[torch.Tensor]] = [left_idepthmap0_prior, left_idepthmap1_prior, left_idepthmap2_prior, left_idepthmap3_prior, left_idepthmap4_raw]
        outputs_left_idepthmap_mask_pyr: List[Optional[torch.Tensor]] = [left_idepthmap0_mask, left_idepthmap1_mask, left_idepthmap2_mask, left_idepthmap3_mask, left_idepthmap4_mask]

        outputs: Dict[str, List[Optional[torch.Tensor]]] = {}
        outputs["left_idepthmap_pyr"] = outputs_left_idepthmap_pyr
        outputs["left_idepthmap_raw_pyr"] = outputs_left_idepthmap_raw_pyr
        outputs["left_idepthmap_mask_pyr"] = outputs_left_idepthmap_mask_pyr

        return outputs
