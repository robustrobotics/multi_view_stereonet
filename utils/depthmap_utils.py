# Copyright 2021 Massachusetts Institute of Technology
#
# @file depthmap_utils.py
# @author W. Nicholas Greene
# @date 2020-02-11 12:35:03 (Tue)

from collections import Counter

import numpy as np

def rectified_disparity_to_depth(fx, B, disparity):
    """Convert a rectified disparity map to depthmap.

    Assumes left/right cameras are rectified.

    fx: Horizontal camera focal length (batch, 1, rows, cols).
    B: Baseline between the left and right camers (batch, 1, rows, cols)
    """
    eps = 1e-7
    depthmap = fx * B / (disparity + eps)
    return depthmap

def depth_to_rectified_disparity(fx, B, depth):
    """Convert a rectified disparity map to depthmap.

    Assumes left/right cameras are rectified.

    fx: Horizontal camera focal length (batch, 1, rows, cols).
    B: Baseline between the left and right camers (batch, 1, rows, cols)
    """
    eps = 1e-7

    disparity = fx * B / (depth + eps)
    return disparity

# def disparity_to_depth(K, Kinv, T, left_disparity):
#     """Convert a disparity map into a depthmap.

#     K: Camera intrinsics of size (batch, 4, 4).
#     T: Pose of left camera in right camera frame of size (batch, 4, 4)
#     """
#     assert(len(K.shape) == 3)
#     assert(len(Kinv.shape) == 3)
#     assert(len(T_left_in_right.shape) == 3)

#     R = T[:, :3, :3]
#     trans = T[:, :3, 3]
#     KRKinv = torch.matmul(K[:, :3, :3], torch.matmul(R * Kinv[:, :3, :3]))

#     KRKinv3 = KRKinv[:, 3, :]
#     Kt = torch.matmul(K[:, :3, :3], trans);

#     w = KRKinv3_(0) * u_ref.x + KRKinv3_(1) * u_ref.y + KRKinv3_(2);
#     Point2s A(w * disparity * epi);
#     Point2s b(Kt_(0) - Kt_(2)*(u_inf.x + disparity * epi.x),
#               Kt_(1) - Kt_(2)*(u_inf.y + disparity * epi.y));

#     Scalar ATA = A.x*A.x + A.y*A.y;
#     Scalar ATb = A.x*b.x + A.y*b.y;

#     POSEST_ASSERT(ATA > 0.0f);

#     return ATb/ATA;

def depthmap_to_point_cloud(K, depthmap):
    """Convert a depthmap to a point cloud.
    """
    assert(K.shape == (3, 3))

    rows = depthmap.shape[0]
    cols = depthmap.shape[1]

    us, vs = np.meshgrid(range(cols), range(rows))

    mask = depthmap.flatten() > 0.0
    mask = mask & ~np.isnan(depthmap.flatten())

    depths = depthmap.flatten()
    depths = depths[mask]

    uvw = np.zeros((rows*cols, 3))
    uvw[:, 0] = us.flatten()
    uvw[:, 1] = vs.flatten()
    uvw[:, 2] = 1.0
    uvw = uvw[mask, :]

    Kinv = np.linalg.inv(K)

    points = np.dot(Kinv, uvw.T).T
    points[:, 0] *= depths
    points[:, 1] *= depths
    points[:, 2] *= depths

    return points

def point_cloud_to_depthmap(shape, P, T, points):
    """Project points onto image plane defined by projection matrix P at pose T.
    """
    assert(P.shape == (4, 4))
    assert(T.shape == (4, 4))
    assert(points.shape[1] == 3)

    Tinv = np.linalg.inv(T)

    points_hom = np.zeros((points.shape[0], 4), np.float32)
    points_hom[:, :3] = points
    points_hom[:, 3] = 1.0

    points_in_cam = np.dot(Tinv, points_hom.T).T

    # Only keep points in front of cam.
    points_in_cam = points_in_cam[points_in_cam[:, 2] > 0, :]

    pixels_in_cam = np.dot(P, points_in_cam.T).T

    pixels = pixels_in_cam[:, :2]
    pixels[:, 0] /= pixels_in_cam[:, 2]
    pixels[:, 1] /= pixels_in_cam[:, 2]

    pixelsi = (pixels + 0.5).astype(np.int32)

    # Check if projections are in bounds.
    mask = pixelsi[:, 0] >= 0.0
    mask = mask & (pixelsi[:, 1] >= 0.0)

    mask = mask & (pixelsi[:, 0] < shape[1])
    mask = mask & (pixelsi[:, 1] < shape[0])

    pixelsi = pixelsi[mask, :]
    pixels = pixels[mask, :]
    depths = points_in_cam[mask, 2]

    # Fill in depths.
    depthmap = np.zeros(shape, dtype=np.float32)
    depthmap[pixelsi[:, 1], pixelsi[:, 0]] = depths

    # # Check for duplicates.
    # lidxs = pixelsi[:, 1] * shape[1] + pixelsi[:, 0]
    # counts = Counter(lidxs)
    # for lidx, count in counts.items():
    #     if count > 1:
    #         mask = lidxs == lidx
    #         col = pixelsi[mask, 0][0]
    #         row = pixelsi[mask, 1][0]
    #         depthmap[row, col] = np.min(depths[mask])

    return depthmap

def depthmap_to_disparity(K, T_right_in_left, depthmap):
    """Convert a depthmap into a disparity map.
    """
    Kinv = np.linalg.inv(K)
    T_left_in_right = np.linalg.inv(T_right_in_left)
    R_left_in_right = T_left_in_right[:3, :3]

    KRKinv = np.dot(K, np.dot(R_left_in_right, Kinv))

    rows = depthmap.shape[0]
    cols = depthmap.shape[1]

    us, vs = np.meshgrid(range(cols), range(rows))

    mask_map = depthmap > 0.0
    mask_map = mask_map & ~np.isnan(depthmap)

    mask = mask_map.flatten()

    uvw = np.zeros((rows*cols, 3))
    uvw[:, 0] = us.flatten()
    uvw[:, 1] = vs.flatten()
    uvw[:, 2] = 1.0
    uvw = uvw[mask, :]

    pix_inf = np.dot(KRKinv, uvw.T).T
    pix_inf[:, 0] /= pix_inf[:, 2]
    pix_inf[:, 1] /= pix_inf[:, 2]

    depths = depthmap.flatten()
    depths = depths[mask]

    points_in_left = np.dot(Kinv, uvw.T).T
    points_in_left[:, 0] *= depths
    points_in_left[:, 1] *= depths
    points_in_left[:, 2] *= depths

    points_in_left_hom = np.zeros((points_in_left.shape[0], 4))
    points_in_left_hom[:, :3] = points_in_left
    points_in_left_hom[:, 3] = 1.0

    points_in_right = np.dot(T_left_in_right, points_in_left_hom.T).T

    pixels_in_right = np.dot(K, points_in_right[:, :3].T).T
    pixels_in_right[:, 0] /= pixels_in_right[:, 2]
    pixels_in_right[:, 1] /= pixels_in_right[:, 2]

    diff = pixels_in_right[:, :2] - pix_inf[:, :2]
    disparities = np.sqrt(np.sum(diff ** 2, 1))

    disparity = np.zeros(depthmap.shape)
    disparity[mask_map] = disparities

    return disparity

def resize_sparse_depthmap(shape_new, Knew, K, depthmap):
    """Resize a depthmap that has invalid values.

    If there are enough invalid values (e.g. zeros), doing a simple image
    resize/interpolation can bias the depths.

    This method projects the depthmap to a pointcloud, then projects the points
    into the new virtual camera to create the resized depthmap.
    """
    points = depthmap_to_point_cloud(K, depthmap)

    P = np.eye(4)
    P[:3, :3] = Knew
    T = np.eye(4)
    depthmap_new = point_cloud_to_depthmap(shape_new, P, T, points)

    return depthmap_new
