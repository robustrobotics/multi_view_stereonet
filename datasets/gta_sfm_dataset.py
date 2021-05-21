# Copyright 2021 Massachusetts Institute of Technology
#
# @file gta_sfm_dataset.py
# @author W. Nicholas Greene
# @date 2020-07-23 13:22:10 (Thu)

import os
import glob

import numpy as np

from datasets import stereo_dataset as sd
from datasets import multi_view_stereo_dataset as mvsd

from utils import depthmap_utils

def sample_comparison_frames_with_poses(
        images, poses, num_comparison_frames,
        min_trans_diff=0.5, max_trans_diff=5.0,
        min_angle_diff_deg=0.0, max_angle_diff_deg=45.0):
    """For each reference frame in list, pick a set of comparison frames.

    Assumes all images from the same sequence.

    Comparison frames will be randomly sampled within the set of frames that are:
      - within [a, b] units in translation
      - within [c, d] degrees in viewing direction (z axis)
    """
    ref_to_cmp = {}
    for ref_idx in range(len(images)):
        ref_pose = poses[ref_idx, :].reshape(4, 4)
        ref_fwd = ref_pose[:3, 2]
        ref_trans = ref_pose[:3, 3]

        # Get list of all comparison frames within bounds.
        valid_cmp_idxs = []
        for cmp_idx in range(len(images)):
            if ref_idx == cmp_idx:
                continue
            cmp_pose = poses[cmp_idx, :].reshape(4, 4)
            cmp_fwd = cmp_pose[:3, 2]
            cmp_trans = cmp_pose[:3, 3]

            # Compute pose difference (translation and viewing direction).
            trans_diff = np.linalg.norm(ref_trans - cmp_trans)
            angle_diff_deg = np.abs(np.arccos(np.dot(ref_fwd, cmp_fwd)) * 180.0 / np.pi)

            if (trans_diff >= min_trans_diff) and (trans_diff <= max_trans_diff) and \
               (angle_diff_deg >= min_angle_diff_deg) and (angle_diff_deg <= max_angle_diff_deg):
                valid_cmp_idxs.append(cmp_idx)

        # Randomly sample.
        if len(valid_cmp_idxs) < num_comparison_frames:
            # No valid comparison frames for this reference frame.
            continue

        valid_cmp_idxs = np.asarray(valid_cmp_idxs)
        permutation = np.random.permutation(np.arange(len(valid_cmp_idxs)))
        sampled_cmp_idxs = valid_cmp_idxs[permutation[:num_comparison_frames]]
        assert(len(sampled_cmp_idxs) == num_comparison_frames)

        ref_to_cmp[images[ref_idx]] = []
        for cmp_idx in range(len(sampled_cmp_idxs)):
            ref_to_cmp[images[ref_idx]].append(images[sampled_cmp_idxs[cmp_idx]])

    return ref_to_cmp

def sample_comparison_frames_with_depthmaps(
        images, depthmaps, Ks, poses, num_comparison_frames,
        min_overlap=0.5, min_trans_diff=0.1):
    """For each reference frame in list, pick a set of comparison frames.

    Assumes all images from the same sequence.

    To sample a comparison frame, we first compute the depth overlap between two
    images. If the overlap is greater than the threshold it will be added to the
    set. The final comparison frame is then sampled from this set.
    """
    pyramid_level = 4

    ref_to_cmp = {}
    for ref_idx in range(len(images)):
        ref_pose = np.copy(poses[ref_idx, :].reshape(4, 4))
        ref_K = np.copy(Ks[ref_idx, :].reshape(3, 3))
        ref_depthmap = np.load(depthmaps[ref_idx])

        # Downsample.
        ref_K /= (1 << pyramid_level)
        ref_K[-1, -1] = 1.0
        ref_K4 = np.eye(4)
        ref_K4[:3, :3] = ref_K
        ref_depthmap = ref_depthmap[::(1 << pyramid_level), ::(1 << pyramid_level)]

        # Get list of all comparison frames within bounds.
        valid_cmp_idxs = []
        valid_cmp_baselines = []
        for cmp_idx in range(len(images)):
            if ref_idx == cmp_idx:
                continue
            cmp_pose = np.copy(poses[cmp_idx, :].reshape(4, 4))
            cmp_K = np.copy(Ks[cmp_idx, :].reshape(3, 3))
            cmp_depthmap = np.load(depthmaps[cmp_idx])

            cmp_K /= 1 << pyramid_level
            cmp_K[-1, -1] = 1.0
            cmp_K4 = np.eye(4)
            cmp_K4[:3, :3] = cmp_K
            cmp_depthmap = cmp_depthmap[::(1 << pyramid_level), ::(1 << pyramid_level)]

            T_ref_in_cmp = np.dot(np.linalg.inv(cmp_pose), ref_pose)
            T_cmp_in_ref = np.dot(np.linalg.inv(ref_pose), cmp_pose)

            trans_diff = np.linalg.norm(T_ref_in_cmp[:3, 3])

            # Compute structure overlap.
            ref_points = depthmap_utils.depthmap_to_point_cloud(ref_K, ref_depthmap)
            ref_to_cmp_depthmap = depthmap_utils.point_cloud_to_depthmap(
                cmp_depthmap.shape, cmp_K4, T_cmp_in_ref, ref_points)
            ref_to_cmp_overlap = np.sum(ref_to_cmp_depthmap > 0) / ref_depthmap.size

            cmp_points = depthmap_utils.depthmap_to_point_cloud(cmp_K, cmp_depthmap)
            cmp_to_ref_depthmap = depthmap_utils.point_cloud_to_depthmap(
                ref_depthmap.shape, ref_K4, T_ref_in_cmp, cmp_points)
            cmp_to_ref_overlap = np.sum(cmp_to_ref_depthmap > 0) / cmp_depthmap.size

            # print("{}, {}, ref_overlap: {:.2f}, cmp_overlap: {:.2f}".format(
            #     ref_idx, cmp_idx, ref_to_cmp_overlap, cmp_to_ref_overlap))

            if (ref_to_cmp_overlap > min_overlap) and (cmp_to_ref_overlap > min_overlap) and \
               (trans_diff > min_trans_diff):
                valid_cmp_idxs.append(cmp_idx)
                valid_cmp_baselines.append(trans_diff)

        # Randomly sample.
        if len(valid_cmp_idxs) < num_comparison_frames:
            # No valid comparison frames for this reference frame.
            continue

        print("image {}: num_cmps: {}".format(ref_idx, len(valid_cmp_idxs)))

        valid_cmp_idxs = np.asarray(valid_cmp_idxs)
        valid_cmp_baselines = np.asarray(valid_cmp_baselines)

        permutation = np.random.permutation(np.arange(len(valid_cmp_idxs)))
        sampled_cmp_idxs = valid_cmp_idxs[permutation[:num_comparison_frames]]
        sampled_cmp_baselines = valid_cmp_baselines[permutation[:num_comparison_frames]]
        assert(len(sampled_cmp_idxs) == num_comparison_frames)

        # Sort by baselines.
        sorted_sampled_cmp_idxs = np.argsort(sampled_cmp_baselines)
        sampled_cmp_idxs = sampled_cmp_idxs[sorted_sampled_cmp_idxs]
        sampled_cmp_baselines = sampled_cmp_baselines[sorted_sampled_cmp_idxs]
        # print(sampled_cmp_baselines)

        ref_to_cmp[images[ref_idx]] = []
        for cmp_idx in range(len(sampled_cmp_idxs)):
            ref_to_cmp[images[ref_idx]].append(images[sampled_cmp_idxs[cmp_idx]])

    return ref_to_cmp

def create_mvs_dataset(data_dir, output_file, num_comparison_frames=1,
                       min_overlap=0.5, ext="jpg", seed=0):
    """Create an MVS dataset from a set of monocular camera trajectories.

    Assumes the input data is organized as follows:
    data_dir
      - sequence0
        - depth
          - <image_num>.npy
        - color
          - <image_num>.jpg
        - poses.txt
        - intrinsics.txt
      - sequence1
      - ...
      - sequenceN

    The selected frames will be written to an output file in the following format:
    <ref_image> <cmp_image0> ... <cmp_imageN>
    <ref_image> <cmp_image0> ... <cmp_imageN>
    """
    np.random.seed(seed)

    assert(not os.path.exists(output_file))

    # Get all sequences in directory.
    sequences = sorted(os.listdir(data_dir))
    # assert(len(sequences) == 200)

    # Loop over each sequence, sample frames, and then write to output file.
    output_stream = open(output_file, "a")
    for sequence in sequences:
        # Grab images.
        images = glob.glob(os.path.join(data_dir, sequence, "color", "*{}".format(ext)))
        images = sorted(images)

        depthmaps = glob.glob(os.path.join(data_dir, sequence, "depth", "*.npy"))
        depthmaps = sorted(depthmaps)

        assert(len(images) == len(depthmaps))
        assert(len(images) > 0)

        print("Sequence: {}, num_images: {}".format(sequence, len(images)))

        # Grab poses.
        pose_data = np.loadtxt(os.path.join(data_dir, sequence, "poses.txt"), skiprows=1, dtype=np.float32)
        pose_ids = pose_data[:, 0]
        poses = pose_data[:, 1:]
        assert(len(pose_ids) == len(images))

        # Grab intrinsics.
        k_data = np.loadtxt(os.path.join(data_dir, sequence, "intrinsics.txt"), skiprows=1, dtype=np.float32)
        k_ids = k_data[:, 0]
        ks = k_data[:, 1:]
        assert(len(k_ids) == len(images))

        # Sample comparison frames.
        ref_to_cmp = sample_comparison_frames_with_depthmaps(
            images, depthmaps, ks, poses, num_comparison_frames, min_overlap)

        # Write to output file.
        for ref_image in images:
            if not ref_image in ref_to_cmp:
                continue

            output_stream.write("{} ".format(os.path.relpath(ref_image, data_dir)))
            for cmp_image in ref_to_cmp[ref_image]:
                output_stream.write("{} ".format(os.path.relpath(cmp_image, data_dir)))
            output_stream.write("\n")

    output_stream.close()

    return

class GTASfMStereoDataset(sd.StereoDataset):
    """Holds stereo data from the GTA-SfM dataset.
    """
    def __init__(self, data_dir, image_file, num_images=0, transform=None,
                 load_groundtruth_depthmaps=False, load_groundtruth_disparity=False):
        super(GTASfMStereoDataset, self).__init__(
            data_dir, image_file, num_images, transform,
            load_groundtruth_depthmaps=load_groundtruth_depthmaps,
            load_groundtruth_disparity=load_groundtruth_disparity)

        self.left_filename_to_idx = {}
        self.right_filename_to_idx = {}
        for idx in range(len(self.left_filenames)):
            self.left_filename_to_idx[os.path.join(self.data_dir, self.left_filenames[idx])] = idx
            self.right_filename_to_idx[os.path.join(self.data_dir, self.right_filenames[idx])]= idx
        
        # Read in calibrations.
        self.left_K = []
        self.right_K = []
        self.left_poses = []
        self.right_poses = []
        for idx in range(len(self.left_filenames)):
            left_tokens = self.left_filenames[idx].split(os.path.sep)
            left_image_id = int(os.path.splitext(left_tokens[-1])[0])

            right_tokens = self.right_filenames[idx].split(os.path.sep)
            right_image_id = int(os.path.splitext(right_tokens[-1])[0])

            # Read in intrinsics.
            K_file = os.path.join(self.data_dir, left_tokens[0], left_tokens[1], "intrinsics.txt")
            K_data = np.loadtxt(K_file, skiprows=1, dtype=np.float32)
            K_ids = K_data[:, 0]
            Ks = K_data[:, 1:]

            left_k_idx = K_ids == left_image_id
            right_k_idx = K_ids == right_image_id

            assert(np.sum(left_k_idx) == 1)
            assert(np.sum(right_k_idx) == 1)

            self.left_K.append(Ks[left_k_idx, :].reshape(3, 3))
            self.right_K.append(Ks[right_k_idx, :].reshape(3, 3))

            # Read in poses.
            pose_file = os.path.join(self.data_dir, left_tokens[0], left_tokens[1], "poses.txt")

            pose_data = np.loadtxt(pose_file, skiprows=1, dtype=np.float32)
            pose_ids = pose_data[:, 0]
            poses = pose_data[:, 1:]

            left_pose_idx = pose_ids == left_image_id
            right_pose_idx = pose_ids == right_image_id

            assert(np.sum(left_pose_idx) == 1)
            assert(np.sum(right_pose_idx) == 1)

            self.left_poses.append(poses[left_pose_idx, :].reshape(4, 4))
            self.right_poses.append(poses[right_pose_idx, :].reshape(4, 4))

        # Fix intrinsics. The principal points for the simulated images appear
        # to be wrong for this dataset. The principal point must lie at the
        # center of the image. For an image of size (rows, cols), this means
        # that cx = (cols-1)/2 and cy = (rows-1)/2, not the naive cols/2 and
        # rows/2, which is was it appears to be.
        for idx in range(len(self.left_K)):
            self.left_K[idx][0, 2] -= 0.5
            self.left_K[idx][1, 2] -= 0.5
            self.right_K[idx][0, 2] -= 0.5
            self.right_K[idx][1, 2] -= 0.5

        return

    def get_calibration(self, idx):
        T_right_in_left = np.dot(np.linalg.inv(self.left_poses[idx]), self.right_poses[idx])
        K3 = self.left_K[idx]
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = K3
        return K, T_right_in_left

    def get_groundtruth_depthmap(self, image_filename):
        """Get groundtruth depthmap for a given image.
        """
        tokens = image_filename.split(os.path.sep)
        tokens[-2] = "depth"
        tokens[-1] = tokens[-1].replace("jpg", "npy")
        depthmap_filename = os.path.join(os.path.sep, *tokens)
        depthmap = np.load(depthmap_filename)
        return depthmap

    def get_groundtruth_disparity(self, image_filename):
        """Get groundtruth disparity for a given image.
        """
        if image_filename in self.left_filename_to_idx:
            image_idx = self.left_filename_to_idx[image_filename]
            K4, T_right_in_left = self.get_calibration(image_idx)
            left_depthmap = self.get_groundtruth_depthmap(image_filename)
            disparity = depthmap_utils.depthmap_to_disparity(K4[:3, :3], T_right_in_left, left_depthmap)
        else:
            image_idx = self.right_filename_to_idx[image_filename]
            K4, T_right_in_left = self.get_calibration(image_idx)
            T_left_in_right = np.linalg.inv(T_right_in_left)
            right_depthmap = self.get_groundtruth_depthmap(image_filename)
            disparity = depthmap_utils.depthmap_to_disparity(K4[:3, :3], T_left_in_right, right_depthmap)

        return disparity

class GTASfMMultiViewStereoDataset(mvsd.MultiViewStereoDataset):
    """Holds stereo data from the GTA-SfM dataset.
    """
    def __init__(self, data_dir, image_file, num_images=0, transform=None,
                 load_groundtruth_depthmaps=False):
        super(GTASfMMultiViewStereoDataset, self).__init__(
            data_dir, image_file, num_images, transform,
            load_groundtruth_depthmaps=load_groundtruth_depthmaps)

        self.left_filename_to_idx = {}
        for idx in range(len(self.left_filenames)):
            self.left_filename_to_idx[os.path.join(self.data_dir, self.left_filenames[idx])] = idx

        # Read in calibrations.
        self.left_K = []
        self.right_K = []
        self.left_poses = []
        self.right_poses = []
        for left_idx in range(len(self.left_filenames)):
            # Read in left data.
            left_tokens = self.left_filenames[left_idx].split(os.path.sep)
            left_image_id = int(os.path.splitext(left_tokens[-1])[0])

            # Read in intrinsics.
            K_file = os.path.join(self.data_dir, left_tokens[0], left_tokens[1], "intrinsics.txt")
            K_data = np.loadtxt(K_file, skiprows=1, dtype=np.float32)
            K_ids = K_data[:, 0]
            Ks = K_data[:, 1:]

            left_k_idx = K_ids == left_image_id
            assert(np.sum(left_k_idx) == 1)
            self.left_K.append(Ks[left_k_idx, :].reshape(3, 3))

            # Read in poses.
            pose_file = os.path.join(self.data_dir, left_tokens[0], left_tokens[1], "poses.txt")

            pose_data = np.loadtxt(pose_file, skiprows=1, dtype=np.float32)
            pose_ids = pose_data[:, 0]
            poses = pose_data[:, 1:]

            left_pose_idx = pose_ids == left_image_id
            assert(np.sum(left_pose_idx) == 1)
            self.left_poses.append(poses[left_pose_idx, :].reshape(4, 4))

            # Read in right data.
            self.right_K.append([])
            self.right_poses.append([])
            for right_idx in range(len(self.right_filenames[left_idx])):
                right_tokens = self.right_filenames[left_idx][right_idx].split(os.path.sep)
                right_image_id = int(os.path.splitext(right_tokens[-1])[0])

                right_k_idx = K_ids == right_image_id
                assert(np.sum(right_k_idx) == 1)
                self.right_K[left_idx].append(Ks[right_k_idx, :].reshape(3, 3))

                right_pose_idx = pose_ids == right_image_id
                assert(np.sum(right_pose_idx) == 1)
                self.right_poses[left_idx].append(poses[right_pose_idx, :].reshape(4, 4))

        # Fix intrinsics. The principal points for the simulated images appear
        # to be wrong for this dataset. The principal point must lie at the
        # center of the image. For an image of size (rows, cols), this means
        # that cx = (cols-1)/2 and cy = (rows-1)/2, not the naive cols/2 and
        # rows/2, which is was it appears to be.
        for left_idx in range(len(self.left_K)):
            self.left_K[left_idx][0, 2] -= 0.5
            self.left_K[left_idx][1, 2] -= 0.5

            for right_idx in range(len(self.right_K[left_idx])):
                self.right_K[left_idx][right_idx][0, 2] -= 0.5
                self.right_K[left_idx][right_idx][1, 2] -= 0.5

        return

    def get_calibration(self, left_idx):
        K3 = self.left_K[left_idx]
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = K3

        T_right_in_left = []
        for right_idx in range(len(self.right_poses[left_idx])):
            T_right_in_left.append(np.dot(np.linalg.inv(self.left_poses[left_idx]), self.right_poses[left_idx][right_idx]))

        return K, T_right_in_left

    def get_groundtruth_depthmap(self, image_filename):
        """Get groundtruth depthmap for a given image.
        """
        tokens = image_filename.split(os.path.sep)
        tokens[-2] = "depth"
        tokens[-1] = tokens[-1].replace("jpg", "npy")
        depthmap_filename = os.path.join(os.path.sep, *tokens)
        depthmap = np.load(depthmap_filename)
        return depthmap
