#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Massachusetts Institute of Technology
#
# @file multi_view_stereo_dataset.py
# @author W. Nicholas Greene
# @date 2021-07-14 13:08:37 (Wed)

import os
import random

import PIL

import torch
import torch.utils.data as tud
import torchvision as tv

import numpy as np

from pyquaternion import Quaternion

from utils import depthmap_utils

def read_images(image_file, replace_jpg_with_png=False):
    """Read in a text file defining a set of stereo pairs.

    Each line of the text file should contain the left and right image filenames:

      left0.jpg right00.jpg ... right0N.jpg
      ...
      leftN.jpg rightM0.jpg ... rightMN.jpg
    """
    left_images = []
    right_images = []
    with open(os.path.join(image_file), "r") as ff:
        lines = ff.readlines()
        for line in lines:
            tokens = line.split()
            left_images.append(tokens[0])
            right_images.append(tokens[1:])

    assert(len(left_images) == len(right_images))

    if replace_jpg_with_png:
        left_images = [image.replace(".jpg", ".png") for image in left_images]
        right_images = [image.replace(".jpg", ".png") for image in right_images]

    return left_images, right_images

def to_tensor(sample):
    return to_tensor_stereo(sample)
def normalize(sample):
    return normalize_stereo(sample)

def get_training_transforms(params):
    """Return transforms to apply to training images.
    """
    if params["augment"]:
        transforms = tv.transforms.Compose([ResizeImageStereo(params["size"][0], params["size"][1]),
                                            RandomColorJitterStereo(),
                                            tv.transforms.Lambda(to_tensor)])
    else:
        transforms = tv.transforms.Compose([ResizeImageStereo(params["size"][0], params["size"][1]),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])

    return transforms

def get_testing_transforms(params, roll_right_image180=False,
                           add_trans_noise=False, add_rot_noise=False):
    """Return transforms to apply to testing images.
    """

    if roll_right_image180:
        roll_right = lambda sample: roll_right_image_180(sample)
        transforms = tv.transforms.Compose([tv.transforms.Lambda(roll_right),
                                            ResizeImageStereo(params["size"][0], params["size"][1]),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])
    elif add_trans_noise:
        trans_noise = lambda sample: add_translation_noise(sample)
        transforms = tv.transforms.Compose([tv.transforms.Lambda(trans_noise),
                                            ResizeImageStereo(params["size"][0], params["size"][1]),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])
    elif add_rot_noise:
        rot_noise = lambda sample: add_rotation_noise(sample)
        transforms = tv.transforms.Compose([tv.transforms.Lambda(rot_noise),
                                            ResizeImageStereo(params["size"][0], params["size"][1]),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])
    else:
        transforms = tv.transforms.Compose([ResizeImageStereo(params["size"][0], params["size"][1]),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])

    return transforms

def normalize_stereo(sample, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Normalize image channels with given mean and std.
    """
    sample["left_image"] = tv.transforms.functional.normalize(sample["left_image"], mean=mean, std=std)
    for idx in range(len(sample["right_image"])):
        sample["right_image"][idx] = tv.transforms.functional.normalize(sample["right_image"][idx], mean=mean, std=std)
    return sample

def to_tensor_stereo(sample):
    """Converts stereo sample to tensors.
    """
    sample["left_image"] = tv.transforms.functional.to_tensor(sample["left_image"])
    for idx in range(len(sample["right_image"])):
        sample["right_image"][idx] = tv.transforms.functional.to_tensor(sample["right_image"][idx])

    sample["K"] = tv.transforms.functional.to_tensor(sample["K"])
    for idx in range(len(sample["T_right_in_left"])):
        sample["T_right_in_left"][idx] = tv.transforms.functional.to_tensor(sample["T_right_in_left"][idx])

    if "left_depthmap_true" in sample:
        sample["left_depthmap_true"] = tv.transforms.functional.to_tensor(sample["left_depthmap_true"])
        for idx in range(len(sample["right_depthmap_true"])):
            sample["right_depthmap_true"][idx] = tv.transforms.functional.to_tensor(sample["right_depthmap_true"][idx])

    return sample

def roll_right_image_180(sample):
    """Roll right image by 180 degrees.
    """
    # Rolling 180 degrees around z axis yields (x, y, z) -> (-x, -y, z).
    T_rolled_in_right = np.eye(4, dtype=np.float32)
    T_rolled_in_right[0, 0] = -1.0
    T_rolled_in_right[1, 1] = -1.0

    for idx in range(len(sample["right_image"])):
        right_image = np.array(sample["right_image"][idx])
        right_rolled = np.copy(np.flipud(np.fliplr(right_image)))
        sample["right_image"][idx] = PIL.Image.fromarray(right_rolled.astype(np.uint8), "RGB")

        T_right_in_left = sample["T_right_in_left"][idx]
        T_rolled_in_left = np.dot(T_right_in_left, T_rolled_in_right)
        sample["T_right_in_left"][idx] = T_rolled_in_left

        if "right_depthmap_true" in sample:
            sample["right_depthmap_true"][idx] = np.copy(np.flipud(np.fliplr(sample["right_depthmap_true"][idx])))

    return sample

def add_translation_noise(sample, sigma=1.0):
    """Add translation noise onto poses.
    """
    for idx in range(len(sample["right_image"])):
        noise = np.random.normal(loc=0.0, scale=sigma, size=(3))
        sample["T_right_in_left"][idx][:3, 3] += noise

    return sample

def add_rotation_noise(sample, sigma_deg=1.0):
    """Add rotation noise onto poses.
    """
    for idx in range(len(sample["right_image"])):
        noise_angle_axis_deg = np.random.normal(loc=0.0, scale=sigma_deg, size=(3))
        noise_angle_deg = np.linalg.norm(noise_angle_axis_deg)
        noise_axis = noise_angle_axis_deg / noise_angle_deg

        q_noise = Quaternion(axis=noise_axis, degrees=noise_angle_deg)
        R_noise = q_noise.rotation_matrix

        R_right_in_left = sample["T_right_in_left"][idx][:3, :3]
        R_right_in_left_noisy = np.dot(R_noise, R_right_in_left)

        sample["T_right_in_left"][idx][:3, :3] = R_right_in_left_noisy

    return sample

class ResizeImageStereo(object):
    """Resize stereo pair.

    Updates intrinsics accordingly.
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        return

    def __call__(self, sample):
        input_rows = sample["left_image"].size[1]
        input_cols = sample["left_image"].size[0]

        # Transform images.
        sample["left_image"] = tv.transforms.functional.resize(
            sample["left_image"], (self.rows, self.cols))

        for idx in range(len(sample["right_image"])):
            sample["right_image"][idx] = tv.transforms.functional.resize(
                sample["right_image"][idx], (self.rows, self.cols))

        # Update intrinsics.
        x_factor = float(self.cols) / input_cols
        y_factor = float(self.rows) / input_rows

        Kold = np.copy(sample["K"])
        sample["K"][0, :] *= x_factor
        sample["K"][1, :] *= y_factor

        # NOTE: Don't resize any groundtruth stuff.

        return sample

class RandomColorJitterStereo(object):
    """Randomly change brightness, contrast, saturation, and hue for a stereo pair.
    """
    def __init__(self, brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                 saturation=(0.8, 1.2), hue=(-0.1, 0.1)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        return

    def __call__(self, sample):
        jitter = tv.transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        sample["left_image"] = jitter(sample["left_image"])
        for idx in range(len(sample["right_image"])):
            sample["right_image"][idx] = jitter(sample["right_image"][idx])
        return sample

class MultiViewStereoDataset(tud.Dataset):
    """Represents a multi-view stereo dataset.

    Assumes fixed intrinsics and extrinsics between images. Designed to be used
    with torch.utils.data.DataLoader.
    """

    def __init__(self, data_dir, image_file, num_images=0, transform=None,
                 load_groundtruth_depthmaps=False):
        """Constructor for StereoDatset base class.

        Takes in a root data directory and a text file of image pairs. Each line
        of the text file should contain the left and right image filenames
        relative to the root directory:

        2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.jpg 2011_09_26/2011_09_26_drive_0002_sync/image_03/data/0000000069.jpg

        :param data_dir: Root directory of images.
        :param image_file: File of image pairs relative to root, one per line.
        :param transform: Transform to apply when reading data.
        """
        super(MultiViewStereoDataset, self).__init__()

        self.data_dir = data_dir
        self.image_file = image_file
        self.transform = transform
        self.load_groundtruth_depthmaps = load_groundtruth_depthmaps

        self.left_filenames, self.right_filenames = read_images(image_file)

        shuffle_on_read = True
        if shuffle_on_read:
            old_num_images = len(self.left_filenames)
            old_left = self.left_filenames
            old_right = self.right_filenames
            permutation = np.random.permutation(old_num_images)
            self.left_filenames = []
            self.right_filenames = []
            for idx in range(old_num_images):
                self.left_filenames.append(old_left[permutation[idx]])
                self.right_filenames.append(old_right[permutation[idx]])

        if num_images > 0:
            # Prune images if desired.
            self.left_filenames = self.left_filenames[0:num_images]
            self.right_filenames = self.right_filenames[0:num_images]

        return

    def get_calibration(self, idx):
        """Must return intrinsics (K) and a list of extrinsics (T_right_in_left) for
        each right image.
        """
        raise NotImplementedError()

    def get_groundtruth_depthmap(self, image_filename):
        """Get groundtruth depthmap for a given image.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read images.
        left_filename = os.path.join(self.data_dir, self.left_filenames[idx])
        assert(os.path.exists(left_filename))
        left_image = PIL.Image.open(left_filename)

        right_filenames = [os.path.join(self.data_dir, right_name) for right_name in self.right_filenames[idx]]
        for right_filename in right_filenames:
            assert(os.path.exists(right_filename))

        right_images = []
        for right_filename in right_filenames:
            right_images.append(PIL.Image.open(right_filename))

        K, T_right_in_left = self.get_calibration(idx)

        sample = {"left_filename": left_filename,
                  "right_filename": right_filename,
                  "left_image": left_image,
                  "right_image": right_images,
                  "K": K,
                  "T_right_in_left": T_right_in_left}

        if self.load_groundtruth_depthmaps:
            sample["left_depthmap_true"] = self.get_groundtruth_depthmap(left_filename)

            sample["right_depthmap_true"] = []
            for right_filename in right_filenames:
                sample["right_depthmap_true"].append(self.get_groundtruth_depthmap(right_filename))

        if self.transform:
            sample = self.transform(sample)

        return sample
