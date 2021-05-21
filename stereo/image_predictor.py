# Copyright 2021 Massachusetts Institute of Technology
#
# @file image_predictor.py
# @author W. Nicholas Greene
# @date 2020-01-24 13:03:36 (Fri)

from typing import List

import numpy as np

import torch
import torch.nn as tnn

import torchvision as tv

from utils import depthmap_utils

def rectified_disparity_to_depth(K, T_right_in_left, left_disparity):
    """Convert a rectified disparity map to depthmap given intrinsics/extrinsics.
    """
    batch_size = K.shape[0]
    rows = left_disparity.shape[-2]
    cols = left_disparity.shape[-1]

    fx = K[:, 0, 0]
    fx_map = fx.view(batch_size, 1, 1, 1).repeat(1, 1, rows, cols)

    baseline = torch.sqrt(torch.sum(T_right_in_left[:, :3, 3]**2, 1))
    baseline_map = baseline.view(batch_size, 1, 1, 1).repeat(1, 1, rows, cols)

    left_depthmap = depthmap_utils.rectified_disparity_to_depth(
        fx_map, baseline_map, left_disparity)

    return left_depthmap

class DepthmapToPointCloud(tnn.Module):
    """Layer to transform a depthmap into a point cloud.

    Based on the implementation from monodepth2.
    """
    def __init__(self):
        super(DepthmapToPointCloud, self).__init__()
        return

    def forward(self, Kinv, depthmap):
        """Returns point cloud in homogeneous coords (xyzw).

        Kinv: Inverse intrinsics matrix of size (batch, 4, 4)
        depthmap: Depthmap of size (batch, 1, rows, cols)
        Returns point cloud of size (batch, 4, rows*cols).
        """
        assert(Kinv.shape[1] == 4)
        assert(Kinv.shape[2] == 4)

        batch_size = depthmap.shape[0]
        rows = depthmap.shape[-2]
        cols = depthmap.shape[-1]

        # Create tensor of homogeneous pixel coordinates of size (batch, 3, rows*cols).
        y_grid, x_grid = torch.meshgrid(torch.arange(0, rows, device=depthmap.device),
                                        torch.arange(0, cols, device=depthmap.device))
        xys = torch.cat([x_grid.reshape(-1, rows * cols).float(),
                         y_grid.reshape(-1, rows * cols).float()], dim=0)
        xys = xys.unsqueeze(0).repeat(batch_size, 1, 1)

        ones = torch.ones(batch_size, 1, rows * cols, dtype=torch.float32, device=xys.device)
        xyz_pix = torch.cat([xys, ones], 1)

        xyz_cam = torch.matmul(Kinv[:, :3, :3], xyz_pix)
        xyz_cam = depthmap.view(batch_size, 1, -1) * xyz_cam
        xyzw_cam = torch.cat([xyz_cam, ones], 1)

        return xyzw_cam

class PointCloudToPixel(tnn.Module):
    """Layer which projects a point cloud into a camera.
    """
    def __init__(self):
        super(PointCloudToPixel, self).__init__()
        return

    def forward(self, K: torch.Tensor, Tinv: torch.Tensor, image_size: List[int], points: torch.Tensor):
        """Returns pixel coordinates in range [-1, 1] to be used by
        torch.nn.function.grid_sample.

        NOTE: This uses the same pixel convention as OpenCV, where (-1, -1)
        corresponds to the top-left corner of the top-left pixel - i.e. *not*
        the pixel center. Given pixel (x, y), the normalized coordinates would
        be:

        x' = 2 (x + 0.5) / cols - 1
        y' = 2 (y + 0.5) / rows - 1

        K: Intrinsics matrix of size (batch, 4, 4)
        Tinv: Inverse of camera pose of size (batch, 4, 4)
        image_size: Image size [rows, cols].
        points: Point cloud of size (batch, 4, rows * cols)
        """
        assert(len(K.shape) == 3)
        assert(len(Tinv.shape) == 3)

        batch_size = K.shape[0]

        P = torch.matmul(K, Tinv)[:, :3, :]
        points_in_cam = torch.matmul(P, points)

        uv = points_in_cam[:, :2, :] / (points_in_cam[:, 2, :].unsqueeze(1) + 1e-7)
        uv = uv.view(batch_size, 2, image_size[0], image_size[1])
        uv = uv.permute(0, 2, 3, 1)

        # See note on pixel convention.
        uv += 0.5
        uv *= 2.0
        uv[..., 0] /= image_size[1]
        uv[..., 1] /= image_size[0]
        uv -= 1.0

        return uv

def disparity_to_idepth(K, T_right_in_left, left_disparity):
    """Function athat transforms general (non-rectified) disparities to inverse
    depths.
    """
    assert(len(T_right_in_left.shape) == 3)
    # assert(T_right_in_left.shape[0] == self.batch_size)
    assert(T_right_in_left.shape[1] == 4)
    assert(T_right_in_left.shape[2] == 4)

    assert(len(K.shape) == 3)
    # assert(K.shape[0] == self.batch_size)
    assert(K.shape[1] == 4)
    assert(K.shape[2] == 4)

    batch_size = K.shape[0]
    rows = left_disparity.shape[-2]
    cols = left_disparity.shape[-1]

    # Create tensor of homogeneous pixel coordinates of size (batch, 3, rows*cols).
    y_grid, x_grid = torch.meshgrid(torch.arange(0, rows, device=left_disparity.device),
                                    torch.arange(0, cols, device=left_disparity.device))
    xys = torch.cat([x_grid.reshape(-1, rows * cols).float(),
                     y_grid.reshape(-1, rows * cols).float()], dim=0)
    xys = xys.unsqueeze(0).repeat(batch_size, 1, 1)

    ones = torch.ones(batch_size, 1, rows * cols, dtype=torch.float32, device=xys.device)
    xyz_pix = torch.cat([xys, ones], 1)

    Kinv = torch.inverse(K)
    T_left_in_right = torch.inverse(T_right_in_left)

    R_left_in_right = T_left_in_right[:, :3, :3]

    KRKinv = torch.matmul(K[:, :3, :3], torch.matmul(R_left_in_right, Kinv[:, :3, :3]))
    KRKinv3 = KRKinv[:, 2, :] # (batch, 3)

    KRKinv3_rep = torch.unsqueeze(KRKinv3, dim=2).repeat(1, 1, rows*cols) # (batch, 3, rows*cols)

    KT_left_in_right = torch.matmul(K, T_left_in_right)
    Kt = KT_left_in_right[:, :3, 3] # (batch, 3)
    Kt_rep = torch.unsqueeze(Kt, dim=2).repeat(1, 1, rows*cols) # (batch, 3, rows*cols)

    # (batch, rows*cols)
    left_disparity_flat = left_disparity.reshape(batch_size, -1)

    # Compute pixels at infinite depth.
    pix_inf = torch.matmul(KRKinv, xyz_pix) # (batch, 3, rows*cols)
    pix_inf[:, 0, :] /= pix_inf[:, 2, :]
    pix_inf[:, 1, :] /= pix_inf[:, 2, :]
    pix_inf[:, 2, :] /= pix_inf[:, 2, :]

    # Compute epipolar lines (must point from far to near depth).
    pix_far = torch.matmul(KRKinv, xyz_pix * 1e2)
    pix_far += Kt_rep
    pix_far[:, 0, :] /= pix_far[:, 2, :]
    pix_far[:, 1, :] /= pix_far[:, 2, :]
    pix_far[:, 2, :] /= pix_far[:, 2, :]

    epi_diff = pix_far[:, :2, :] - pix_inf[:, :2, :]
    epi_norm = torch.sqrt(torch.sum(epi_diff**2, dim=1))
    epiline = epi_diff[:, :2, :] # (batch, 2, rows*cols)
    epiline[:, 0, :] /= (epi_norm + 1e-6)
    epiline[:, 1, :] /= (epi_norm + 1e-6)

    mask = epi_norm < 1e-6
    mask = mask.reshape(batch_size, 1, rows, cols)

    # Convert disparity to idepth.
    # (batch, rows*cols)
    w = KRKinv3_rep[:, 0, :] * xyz_pix[:, 0, :] + \
        KRKinv3_rep[:, 1, :] * xyz_pix[:, 1, :] + \
        KRKinv3_rep[:, 2, :]

    # (batch, rows*cols)
    A0 = Kt_rep[:, 0, :] - Kt_rep[:, 2, :]*(pix_inf[:, 0, :] + left_disparity_flat * epiline[:, 0, :])
    A1 = Kt_rep[:, 1, :] - Kt_rep[:, 2, :]*(pix_inf[:, 1, :] + left_disparity_flat * epiline[:, 1, :])

    b0 = w * left_disparity_flat * epiline[:, 0, :]
    b1 = w * left_disparity_flat * epiline[:, 1, :]

    ATA = A0 * A0 + A1 * A1
    ATb = A0 * b0 + A1 * b1

    left_idepthmap = ATb / ATA
    left_idepthmap = left_idepthmap.reshape(batch_size, 1, rows, cols)

    # Set bad points to 0 idepth.
    left_idepthmap = (~mask).float() * left_idepthmap

    return left_idepthmap

class DisparityToIDepth(tnn.Module):
    """Layer to transform a (non-rectified) disparities to inverse depths.
    """
    def __init__(self):
        super(DisparityToIDepth, self).__init__()
        return

    def forward(self, K, T_right_in_left, left_disparity):
        """Converts (non-rectified) disparities to inverse depths.
        """
        return disparity_to_idepth(K, T_right_in_left, left_disparity)

class IDepthToDisparity(tnn.Module):
    """Layer to transform an inverse depthmap to (non-rectified) disparities.
    """
    def __init__(self):
        super(IDepthToDisparity, self).__init__()
        self.depthmap_to_pointcloud = DepthmapToPointCloud()
        return

    def forward(self, K, T_right_in_left, left_idepthmap):
        """Converts inverse depths to (non-rectified) disparities.
        """
        assert(len(T_right_in_left.shape) == 3)
        # assert(T_right_in_left.shape[0] == self.batch_size)
        assert(T_right_in_left.shape[1] == 4)
        assert(T_right_in_left.shape[2] == 4)

        assert(len(K.shape) == 3)
        # assert(K.shape[0] == self.batch_size)
        assert(K.shape[1] == 4)
        assert(K.shape[2] == 4)

        batch_size = K.shape[0]
        rows = left_idepthmap.shape[-2]
        cols = left_idepthmap.shape[-1]

        # Create tensor of homogeneous pixel coordinates of size (batch, 3, rows*cols).
        y_grid, x_grid = torch.meshgrid(torch.arange(0, rows, device=left_idepthmap.device),
                                        torch.arange(0, cols, device=left_idepthmap.device))
        xys = torch.cat([x_grid.reshape(-1, rows * cols).float(),
                         y_grid.reshape(-1, rows * cols).float()], dim=0)
        xys = xys.unsqueeze(0).repeat(batch_size, 1, 1)

        ones = torch.ones(batch_size, 1, rows * cols, dtype=torch.float32, device=xys.device)
        xyz_pix = torch.cat([xys, ones], 1)

        Kinv = torch.inverse(K)
        T_left_in_right = torch.inverse(T_right_in_left)
        R_left_in_right = T_left_in_right[:, :3, :3]

        KRKinv = torch.matmul(K[:, :3, :3], torch.matmul(R_left_in_right, Kinv[:, :3, :3]))

        # Compute pixels at infinite depth.
        pix_inf = torch.matmul(KRKinv, xyz_pix) # (batch, 3, rows*cols)
        pix_inf[:, 0, :] /= pix_inf[:, 2, :]
        pix_inf[:, 1, :] /= pix_inf[:, 2, :]
        pix_inf[:, 2, :] /= pix_inf[:, 2, :]

        # Compute projection.
        # Convert idepths to depths.
        left_depthmap = 1.0 / (left_idepthmap + 1e-6)

        # Convert depth to point cloud.
        left_points = self.depthmap_to_pointcloud(Kinv, left_depthmap)

        # Transform point cloud to right frame to generate idepths.
        right_points = torch.matmul(T_left_in_right[:, :3, :], left_points)
        right_pixels = torch.matmul(K[:, :3, :3], right_points)
        right_pixw = right_pixels[:, 2, :].clone() # Needed to avoid inplace op.
        right_pixels[:, 0, :] /= right_pixw
        right_pixels[:, 1, :] /= right_pixw

        left_disparity = torch.norm(right_pixels[:, :2, :] - pix_inf[:, :2, :], dim=1)
        left_disparity = left_disparity.reshape(batch_size, 1, rows, cols)

        return left_disparity

class RectifiedImagePredictor(tnn.Module):
    """Predicts a given image using a source image and a rectified disparity map.
    """
    def __init__(self):
        super(RectifiedImagePredictor, self).__init__()
        return

    def forward(self, K, T_right_in_left, left_disparity, right_image):
        """Predict the left image given the right image and a left disparity map
        (defined in pixels).
        """
        assert(len(T_right_in_left.shape) == 3)
        # assert(T_right_in_left.shape[0] == self.batch_size)
        assert(T_right_in_left.shape[1] == 4)
        assert(T_right_in_left.shape[2] == 4)

        assert(len(K.shape) == 3)
        # assert(K.shape[0] == self.batch_size)
        assert(K.shape[1] == 4)
        assert(K.shape[2] == 4)

        batch_size = left_disparity.shape[0]
        rows = left_disparity.shape[-2]
        cols = left_disparity.shape[-1]

        # Create tensor of infinite pixel coordinates of size (batch, rows, cols, 2).
        y_grid, x_grid = torch.meshgrid(torch.arange(0, rows, device=right_image.device),
                                        torch.arange(0, cols, device=right_image.device))
        xys = torch.cat([x_grid.unsqueeze(0).float(), y_grid.unsqueeze(0).float()], dim=0) # (2, rows, cols)
        xys = xys.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (batch, 2, rows, cols)

        pix_inf = xys.permute(0, 2, 3, 1) # (batch, rows, cols, 2)

        sign = torch.sign(T_right_in_left[:, 0, 3])
        sign = sign.unsqueeze(dim=1).unsqueeze(dim=2)
        sign = sign.repeat(1, rows, cols)

        left_pixels = pix_inf
        left_pixels[..., 0] = left_pixels[..., 0] - sign * torch.squeeze(left_disparity)

        # Normalize pixels to work with grid_sample. (-1, -1) corresponds to the
        # top-left corner of the top-left pixel - i.e. *not* the pixel
        # center. Given pixel (x, y), the normalized coordinates would be:
        # x' = 2 (x + 0.5) / cols - 1
        # y' = 2 (y + 0.5) / rows - 1
        left_pixels = left_pixels + 0.5
        left_pixels = 2.0 * left_pixels
        left_pixels[..., 0] /= cols
        left_pixels[..., 1] /= rows
        left_pixels -= 1.0

        # mask = torch.ones(batch_size, rows, cols, dtype=torch.uint8)
        xmask = torch.abs(left_pixels[:, :, :, 0]) > 1.0
        ymask = torch.abs(left_pixels[:, :, :, 1]) > 1.0
        mask = xmask | ymask
        mask = torch.unsqueeze(mask, 1)

        # Construct image given pixel map.
        left_image_pred = tnn.functional.grid_sample(
            right_image, left_pixels, mode="bilinear",
            padding_mode="border", align_corners=False)

        return left_image_pred, mask

class IDepthImagePredictor(tnn.Module):
    """Predicts a given image using a source image and an inverse depthmap.
    """
    def __init__(self):
        super(IDepthImagePredictor, self).__init__()
        self.depthmap_to_pointcloud = DepthmapToPointCloud()
        self.pointcloud_to_pixel = PointCloudToPixel()
        return

    def forward(self, K, T_right_in_left, left_idepthmap, right_image):
        """Predict the left image given the right image and a left disparity map.
        """
        assert(len(T_right_in_left.shape) == 3)
        # assert(T_right_in_left.shape[0] == self.batch_size)
        assert(T_right_in_left.shape[1] == 4)
        assert(T_right_in_left.shape[2] == 4)

        assert(len(K.shape) == 3)
        # assert(K.shape[0] == self.batch_size)
        assert(K.shape[1] == 4)
        assert(K.shape[2] == 4)

        Kinv = torch.inverse(K)
        T_left_in_right = torch.inverse(T_right_in_left)

        # Assume that idepths are > 0.
        left_depthmap = 1.0 / (left_idepthmap + 1e-6)

        # Convert depth to point cloud.
        left_points = self.depthmap_to_pointcloud(Kinv, left_depthmap)

        # Convert point cloud to pixels in the right image.
        left_pixels = self.pointcloud_to_pixel(K, T_left_in_right, right_image.shape[-2:], left_points)

        # mask = torch.ones(self.batch_size, self.rows, self.cols, dtype=torch.uint8)
        xmask = torch.abs(left_pixels[:, :, :, 0]) > 1.0
        ymask = torch.abs(left_pixels[:, :, :, 1]) > 1.0
        mask = xmask | ymask
        mask = torch.unsqueeze(mask, 1)

        # Construct image given pixel map.
        left_image_pred = tnn.functional.grid_sample(
            right_image, left_pixels, mode="bilinear",
            padding_mode="border", align_corners=False)

        return left_image_pred, mask

def get_fronto_parallel_homography(K_left, K_right, T_left_in_right, idepth):
    """Compute the homography that transfer pixels from the left camera to the right
    camera assuming a fronto-parallel plane at the given idepth away from the
    left camera.

    Derivation:

    Assume we are given a left and right camera with intrinsics K_l and K_r and
    relative pose T_r_in_l (or equivalently T_l_in_r).

    A point x will be denoted x_l in the left coordinate system and x_r in the
    right coordinate system. We then have (ignoring homogenous coords to
    simplify notation)

    x_r = T_l_in_r * x_l
        = R_l_in_r * x_l + t_l_in_r

    Now suppose there exists a plane with (outward) normal n_l and depth d_l
    defined in the left coordinate system such that n_l^T * x_l - d_l =
    0. Rearranging terms we have

    1 = (n_l^T * x_l) / d_l

    Then t_l_in_r = t_l_in_r * (n_l^T * x_l) / d_l and we have

    x_r = R_l_in_r * x_l + t_l_in_r * (n_l^T * x_l) / d_l
        = (R_l_in_r + t_l_in_r * n_l^T / d_l) * x_l

    Finally replacing x_r by K_r_inv * u_r and x_l by K_l_inv * u_l we have

    K_r_inv * u_r \propto (R_l_in_r + t_l_in_r * n_l^T / d_l) * K_l_inv * u_l
    u_r \propto K_r * (R_l_in_r + t_l_in_r * n_l^T / d_l) * K_l_inv * u_l
    u_r \propto H_l_in_r * u_l

    where H_l_in_r = K_r * (R_l_in_r + t_l_in_r * n_l^T / d_l) * K_l_inv.
    """
    batch_size = K_left.shape[0]

    assert(K_left.shape[1] == 3)
    assert(K_left.shape[2] == 3)
    assert(K_right.shape[1] == 3)
    assert(K_right.shape[2] == 3)
    assert(T_left_in_right.shape[1] == 4)
    assert(T_left_in_right.shape[2] == 4)
    assert(idepth.shape[0] == batch_size)

    R = T_left_in_right[:, :3, :3]
    trans = T_left_in_right[:, :3, 3]

    trans_idepth = trans * idepth.unsqueeze(1).repeat(1, 3)

    trans_idepth_nT = torch.zeros(R.shape, device=R.device)
    trans_idepth_nT[:, :, 2] = trans_idepth

    H_left_in_right = R + trans_idepth_nT

    K_left_inv = torch.inverse(K_left)

    H_left_in_right = torch.matmul(H_left_in_right, K_left_inv)
    H_left_in_right = torch.matmul(K_right, H_left_in_right)

    return H_left_in_right

class HomographyImagePredictor(tnn.Module):
    """Predicts a given image using a source image and a homography.
    """
    def __init__(self):
        super(HomographyImagePredictor, self).__init__()
        return

    def forward(self, H_left_in_right, right_image):
        """Predict the left image given the right image and homography.
        """
        assert(len(H_left_in_right.shape) == 3)
        # assert(H_left_in_right.shape[0] == self.batch_size)
        assert(H_left_in_right.shape[1] == 3)
        assert(H_left_in_right.shape[2] == 3)

        batch_size = right_image.shape[0]
        rows = right_image.shape[-2]
        cols = right_image.shape[-1]

        # Create tensor of homogeneous pixel coordinates of size (batch, 3, rows*cols).
        y_grid, x_grid = torch.meshgrid(torch.arange(0, rows, device=right_image.device),
                                        torch.arange(0, cols, device=right_image.device))
        xys = torch.cat([x_grid.reshape(-1, rows * cols).float(),
                         y_grid.reshape(-1, rows * cols).float()], dim=0)
        xys = xys.unsqueeze(0).repeat(batch_size, 1, 1)

        ones = torch.ones(batch_size, 1, rows * cols, dtype=torch.float32, device=xys.device)
        xyz_left = torch.cat([xys, ones], 1)

        # Apply homography to map pixels from left to right image.
        xyz_right = torch.matmul(H_left_in_right, xyz_left)
        right_pixels = xyz_right[:, 0:2, :] # (batch, 2, rows*cols)
        right_pixels[:, 0, :] /= xyz_right[:, 2, :]
        right_pixels[:, 1, :] /= xyz_right[:, 2, :]

        # Normalize pixels to work with grid_sample. (-1, -1) corresponds to the
        # top-left corner of the top-left pixel - i.e. *not* the pixel
        # center. Given pixel (x, y), the normalized coordinates would be:
        # x' = 2 (x + 0.5) / cols - 1
        # y' = 2 (y + 0.5) / rows - 1
        right_pixels = right_pixels.reshape(batch_size, 2, rows, cols)
        right_pixels = right_pixels.permute(0, 2, 3, 1) # (batch, rows, cols, 2)

        right_pixels = right_pixels + 0.5
        right_pixels = 2.0 * right_pixels
        right_pixels[..., 0] /= cols
        right_pixels[..., 1] /= rows
        right_pixels -= 1.0

        # mask = torch.ones(self.batch_size, self.rows, self.cols, dtype=torch.uint8)
        xmask = torch.abs(right_pixels[:, :, :, 0]) > 1.0
        ymask = torch.abs(right_pixels[:, :, :, 1]) > 1.0
        mask = xmask | ymask
        mask = torch.unsqueeze(mask, 1)

        # Construct image given pixel map.
        left_image_pred = tnn.functional.grid_sample(
            right_image, right_pixels, mode="bilinear",
            padding_mode="border", align_corners=False)

        return left_image_pred, mask

class IDepthmapProjector(tnn.Module):
    """Projects a left inverse depthmap to the right frame, generating right pixels
    and idepths.

    Does not assume rectified images (i.e. can be used for motion-stereo).
    """

    def __init__(self):
        super(IDepthmapProjector, self).__init__()
        self.depthmap_to_pointcloud = DepthmapToPointCloud()
        self.pointcloud_to_pixel = PointCloudToPixel()
        return

    def forward(self, K, T_right_in_left, left_idepthmap):
        """Projects the left_idepthmap to the right frame.
        """
        assert(len(T_right_in_left.shape) == 3)
        # assert(T_right_in_left.shape[0] == self.batch_size)
        assert(T_right_in_left.shape[1] == 4)
        assert(T_right_in_left.shape[2] == 4)

        assert(len(K.shape) == 3)
        # assert(K.shape[0] == self.batch_size)
        assert(K.shape[1] == 4)
        assert(K.shape[2] == 4)

        batch_size = K.shape[0]

        Kinv = torch.inverse(K)
        T_left_in_right = torch.inverse(T_right_in_left)

        # Convert idepths to depths.
        left_depthmap = 1.0 / (left_idepthmap + 1e-6)

        # Convert depth to point cloud.
        left_points = self.depthmap_to_pointcloud(Kinv, left_depthmap)

        # Transform point cloud to right frame to generate idepths.
        right_points = torch.matmul(T_left_in_right[:, :3, :], left_points)
        right_idepths = 1.0 / (right_points[:, 2, :] + 1e-6)
        right_idepths = right_idepths.view(left_idepthmap.shape)

        # Convert left point cloud to pixels in the right image.
        right_pixels = self.pointcloud_to_pixel(K, T_left_in_right, left_idepthmap.shape[-2:], left_points)

        # mask = torch.ones(self.batch_size, self.rows, self.cols, dtype=torch.uint8)
        xmask = torch.abs(right_pixels[:, :, :, 0]) > 1.0
        ymask = torch.abs(right_pixels[:, :, :, 1]) > 1.0
        mask = xmask | ymask
        mask = torch.unsqueeze(mask, 1)

        return right_pixels, right_idepths, mask

class ImagePredictor(tnn.Module):
    """Predicts a given image using a source image and a disparity map.
    """
    def __init__(self):
        super(ImagePredictor, self).__init__()
        self.disparity_to_idepth = DisparityToIDepth()
        self.idepthmap_projector = IDepthmapProjector()
        return

    def forward(self, K, T_right_in_left, left_disparity, right_image):
        """Predict the left image given the right image and a left disparity map.
        """
        # Convert disparities to idepths.
        left_idepthmap = self.disparity_to_idepth(K, T_right_in_left, left_disparity)

        # Project idepths to right frame.
        right_pixels, right_idepths, mask = self.idepthmap_projector(K, T_right_in_left, left_idepthmap)

        # Construct image given pixel map.
        left_image_pred = tnn.functional.grid_sample(
            right_image, right_pixels, mode="bilinear",
            padding_mode="border", align_corners=False)

        return left_image_pred, mask
