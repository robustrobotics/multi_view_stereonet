#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Massachusetts Institute of Technology
#
# @file stereo_dataset.py
# @author W. Nicholas Greene
# @date 2021-07-14 13:08:50 (Wed)

import os
import random

import PIL

import torch
import torch.utils.data as tud
import torchvision as tv

import numpy as np

from utils import depthmap_utils

def read_stereo_pairs(image_file, replace_jpg_with_png=False):
    """Read in a text file defining a set of stereo pairs.

    Each line of the text file should contain the left and right image filenames:

      left0.jpg right0.jpg
      ...
      leftN.jpg rightN.jpg
    """
    left_images = []
    right_images = []
    with open(os.path.join(image_file), "r") as ff:
        lines = ff.readlines()
        for line in lines:
            tokens = line.split()
            left_images.append(tokens[0])
            right_images.append(tokens[1])

    assert(len(left_images) == len(right_images))

    if replace_jpg_with_png:
        left_images = [image.replace(".jpg", ".png") for image in left_images]
        right_images = [image.replace(".jpg", ".png") for image in right_images]

    return left_images, right_images

def horizontal_flip_stereo(left, right):
    """Perform a horizontal flip on a stereo pair.

    Note that when flipping, we need to switch the roles of the left/right
    image.
    """
    flipped_left = tv.transforms.functional.hflip(right)
    flipped_right = tv.transforms.functional.hflip(left)
    return flipped_left, flipped_right

def to_tensor(sample):
    return to_tensor_stereo(sample)
def normalize(sample):
    return normalize_stereo(sample)

def get_training_transforms(params):
    """Return transforms to apply to training images.
    """
    if params["augment"]:
        transforms = tv.transforms.Compose([ResizeImageStereo(params["size"][0], params["size"][1]),
                                            # RandomHorizontalFlipStereo(), # Not sure if this should switch filenames as well. Turn off for now.
                                            RandomColorJitterStereo(),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])
    else:
        transforms = tv.transforms.Compose([ResizeImageStereo(params["size"][0], params["size"][1]),
                                            tv.transforms.Lambda(to_tensor),
                                            tv.transforms.Lambda(normalize)])

    return transforms

def get_testing_transforms(params, roll_right_image180=False):
    """Return transforms to apply to testing images.
    """
    if roll_right_image180:
        roll_right = lambda sample: roll_right_image_180(sample)
        transforms = tv.transforms.Compose([tv.transforms.Lambda(roll_right),
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
    sample["right_image"] = tv.transforms.functional.normalize(sample["right_image"], mean=mean, std=std)
    return sample

def to_tensor_stereo(sample):
    """Converts stereo sample to tensors.
    """
    sample["left_image"] = tv.transforms.functional.to_tensor(sample["left_image"])
    sample["right_image"] = tv.transforms.functional.to_tensor(sample["right_image"])

    sample["K"] = tv.transforms.functional.to_tensor(sample["K"])
    sample["T_right_in_left"] = tv.transforms.functional.to_tensor(sample["T_right_in_left"])

    if "left_disparity_true" in sample:
        sample["left_disparity_true"] = tv.transforms.functional.to_tensor(sample["left_disparity_true"])
        sample["right_disparity_true"] = tv.transforms.functional.to_tensor(sample["right_disparity_true"])

    if "left_depthmap_true" in sample:
        sample["left_depthmap_true"] = tv.transforms.functional.to_tensor(sample["left_depthmap_true"])
        sample["right_depthmap_true"] = tv.transforms.functional.to_tensor(sample["right_depthmap_true"])

    return sample

def roll_right_image_180(sample):
    """Roll right image by 180 degrees.
    """
    right_image = np.array(sample["right_image"])
    right_rolled = np.copy(np.flipud(np.fliplr(right_image)))
    sample["right_image"] = PIL.Image.fromarray(right_rolled.astype(np.uint8), "RGB")

    # Rolling 180 degrees around z axis yields (x, y, z) -> (-x, -y, z).
    T_rolled_in_right = np.eye(4, dtype=np.float32)
    T_rolled_in_right[0, 0] = -1.0
    T_rolled_in_right[1, 1] = -1.0

    T_right_in_left = sample["T_right_in_left"]
    T_rolled_in_left = np.dot(T_right_in_left, T_rolled_in_right)
    sample["T_right_in_left"] = T_rolled_in_left

    if "right_disparity_true" in sample:
        sample["right_disparity_true"] = np.copy(np.flipud(np.fliplr(sample["right_disparity_true"])))

    if "right_depthmap_true" in sample:
        sample["right_depthmap_true"] = np.copy(np.flipud(np.fliplr(sample["right_depthmap_true"])))

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
        sample["right_image"] = tv.transforms.functional.resize(
            sample["right_image"], (self.rows, self.cols))

        # Update intrinsics.
        x_factor = float(self.cols) / input_cols
        y_factor = float(self.rows) / input_rows

        Kold = np.copy(sample["K"])
        sample["K"][0, :] *= x_factor
        sample["K"][1, :] *= y_factor

        # NOTE: Don't resize any groundtruth stuff.

        return sample

class RandomHorizontalFlipStereo(object):
    """Perform a horizontal flip on a stereo pair.

    Mathematically, this is a reflection across the YZ plane centered at the
    left camera (i.e. (x, y, z) -> (-x, y, z)).
    """
    def __init__(self, prob=0.5):
        self.prob = prob

        self.reflection = np.eye(4, dtype=np.float32)
        self.reflection[0, 0] = -1.0

        return

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["left_image"], sample["right_image"] = horizontal_flip_stereo(
                sample["left_image"], sample["right_image"])

            # Reflect pose and then fix x-axis to point right again.
            sample["T_right_in_left"] = np.dot(
                self.reflection, np.linalg.inv(sample["T_right_in_left"]))
            sample["T_right_in_left"][:3, 0] = np.cross(
                sample["T_right_in_left"][:3, 1], sample["T_right_in_left"][:3, 2])

            if "left_disparity_true" in sample:
                sample["left_disparity_true"], sample["right_disparity_true"] = horizontal_flip_stereo(
                    sample["left_disparity_true"], sample["right_disparity_true"])

            if "left_depthmap_true" in sample:
                sample["left_depthmap_true"], sample["right_depthmap_true"] = horizontal_flip_stereo(
                    sample["left_depthmap_true"], sample["right_depthmap_true"])

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
        sample["right_image"] = jitter(sample["right_image"])
        return sample

class StereoDataset(tud.Dataset):
    """Represents stereo dataset.

    Assumes fixed intrinsics and extrinsics between images. Designed to be used
    with torch.utils.data.DataLoader.

    For now assumes data is in KITTI raw format.
    """

    def __init__(self, data_dir, image_file, num_images=0, transform=None,
                 load_groundtruth_depthmaps=False, load_groundtruth_disparity=False):
        """Constructor for StereoDatset base class.

        Takes in a root data directory and a text file of image pairs. Each line
        of the text file should contain the left and right image filenames
        relative to the root directory:

        2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.jpg 2011_09_26/2011_09_26_drive_0002_sync/image_03/data/0000000069.jpg

        :param data_dir: Root directory of images.
        :param image_file: File of image pairs relative to root, one per line.
        :param transform: Transform to apply when reading data.
        """
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.image_file = image_file
        self.transform = transform
        self.load_groundtruth_depthmaps = load_groundtruth_depthmaps
        self.load_groundtruth_disparity = load_groundtruth_disparity

        self.left_filenames, self.right_filenames = read_stereo_pairs(image_file)

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
        """Must return intrinsics (K) and extrinsics (T_right_in_left).
        """
        raise NotImplementedError()

    def get_groundtruth_depthmap(self, image_filename):
        """Get groundtruth depthmap for a given image.
        """
        raise NotImplementedError()

    def get_groundtruth_disparity(self, image_filename):
        """Get groundtruth disparity for a given image.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read images.
        left_filename = os.path.join(self.data_dir, self.left_filenames[idx])
        right_filename = os.path.join(self.data_dir, self.right_filenames[idx])

        assert(os.path.exists(left_filename))
        assert(os.path.exists(right_filename))

        left_image = PIL.Image.open(left_filename)
        right_image = PIL.Image.open(right_filename)

        K, T_right_in_left = self.get_calibration(idx)

        sample = {"left_filename": left_filename,
                  "right_filename": right_filename,
                  "left_image": left_image,
                  "right_image": right_image,
                  "K": K,
                  "T_right_in_left": T_right_in_left}

        if self.load_groundtruth_disparity:
            sample["left_disparity_true"] = self.get_groundtruth_disparity(left_filename)
            sample["right_disparity_true"] = self.get_groundtruth_disparity(right_filename)

        if self.load_groundtruth_depthmaps:
            sample["left_depthmap_true"] = self.get_groundtruth_depthmap(left_filename)
            sample["right_depthmap_true"] = self.get_groundtruth_depthmap(right_filename)

        if self.transform:
            sample = self.transform(sample)

        return sample
