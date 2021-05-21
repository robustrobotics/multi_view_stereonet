# Copyright 2021 Massachusetts Institute of Technology
#
# @file demon_dataset.py
# @author W. Nicholas Greene
# @date 2020-10-09 11:55:58 (Fri)

import os
import glob
import random

from PIL import Image
import numpy as np

import torch.utils.data as tud

from utils import depthmap_utils

class DeMoNDataset(tud.Dataset):
    """Dataset class for DeMoN data based on the implementation in DPSNet.

    Assumes data is organized as follows:
       root/scene_1/0000000.jpg
       root/scene_1/0000001.jpg
       ...
       root/scene_1/cam.txt
       root/scene_1/poses.txt
       root/scene_2/0000000.jpg
       root/scene_2/0000001.jpg
       ...
       root/scene_2/cam.txt
       root/scene_2/poses.txt
       ...

    The train text file should list each scene desired:
       scene_1
       scene_2
       ...
    """

    def __init__(self, data_dir, input_file, num_right_images=1, num_left_images=0, transform=None):
        self.data_dir = data_dir
        self.input_file = input_file
        self.num_right_images = num_right_images
        self.num_left_images = num_left_images
        self.transform = transform

        # Get scenes.
        scenes = []
        with open(os.path.join(self.data_dir, self.input_file), "r") as stream:
            scenes = stream.readlines()
        scenes = [os.path.join(self.data_dir, scene.strip()) for scene in scenes]
        self.scenes = sorted(scenes)

        # Generate samples.
        self.samples = self.generate_samples(self.num_right_images)

        shuffle_on_read = True
        if shuffle_on_read:
            random.shuffle(self.samples)

        if self.num_left_images > 0:
            self.samples = self.samples[:self.num_left_images]

        self.left_filename_to_idx = {}
        for idx in range(len(self.samples)):
            self.left_filename_to_idx[self.samples[idx]["left_filename"]] = idx

        return

    def generate_samples(self, num_right_images):
        samples = []
        demi_length = (num_right_images + 1) // 2

        for scene in self.scenes:
            assert(os.path.exists(os.path.join(scene, "cam.txt")))
            assert(os.path.exists(os.path.join(scene, "poses.txt")))

            K3 = np.genfromtxt(os.path.join(scene, "cam.txt")).astype(np.float32).reshape((3, 3))
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = K3

            inv_poses = np.genfromtxt(os.path.join(scene, "poses.txt")).astype(np.float32)
            images = sorted(glob.glob(os.path.join(scene, "*.jpg")))

            if len(images) < num_right_images + 1:
                continue

            for left_idx in range(len(images)):
                if left_idx < demi_length:
                    shifts = list(range(0, num_right_images + 1))
                    shifts.pop(left_idx)
                elif left_idx >= len(images) - demi_length:
                    shifts = list(range(len(images) - (num_right_images + 1), len(images)))
                    shifts.pop(left_idx - len(images))
                else:
                    shifts = list(range(left_idx - demi_length, left_idx + (num_right_images + 1 + 1) // 2))
                    shifts.pop(demi_length)

                assert(len(shifts) == num_right_images)

                left_filename = images[left_idx]
                left_depthmap_true_filename = os.path.splitext(left_filename)[0] + ".npy"
                T_world_in_left = np.concatenate((inv_poses[left_idx, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)

                sample = {"K": K,
                          "left_filename": left_filename,
                          "left_depthmap_true_filename": left_depthmap_true_filename,
                          "right_filename": [],
                          "right_depthmap_true_filename": [],
                          "T_right_in_left": []}

                for right_idx in shifts:
                    right_filename = images[right_idx]
                    sample["right_filename"].append(right_filename)
                    sample["right_depthmap_true_filename"].append(os.path.splitext(right_filename)[0] + ".npy")
                    assert(os.path.exists(sample["right_depthmap_true_filename"][-1]))

                    T_world_in_right = np.concatenate((inv_poses[right_idx, :].reshape((3, 4)), np.array([[0, 0, 0, 1]])), axis=0)
                    T_right_in_left = T_world_in_left @ np.linalg.inv(T_world_in_right)
                    T_right_in_left = T_right_in_left.astype(np.float32)
                    sample["T_right_in_left"].append(T_right_in_left)

                assert(len(sample["right_filename"]) == num_right_images)
                assert(len(sample["T_right_in_left"]) == num_right_images)

                samples.append(sample)

        return samples

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        # Read in images.
        left_image = Image.open(raw_sample["left_filename"])
        left_depthmap_true = np.load(raw_sample["left_depthmap_true_filename"]).astype(np.float32)

        right_images = []
        right_depthmap_true = []
        for idx in range(len(raw_sample["right_filename"])):
            right_images.append(Image.open(raw_sample["right_filename"][idx]))
            right_depthmap_true.append(np.load(raw_sample["right_depthmap_true_filename"][idx]).astype(np.float32))

        sample = {"left_filename": raw_sample["left_filename"],
                  "right_filename": raw_sample["right_filename"],
                  "left_image": left_image,
                  "right_image": right_images,
                  "K": raw_sample["K"],
                  "T_right_in_left": raw_sample["T_right_in_left"],
                  "left_depthmap_true": left_depthmap_true,
                  "right_depthmap_true": right_depthmap_true}

        assert(len(sample["right_filename"]) == self.num_right_images)
        assert(len(sample["T_right_in_left"]) == self.num_right_images)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

class DeMoNStereoDataset(tud.Dataset):
    """Wrapper around DeMoNDataset for stereo.
    """
    def __init__(self, data_dir, input_file, num_left_images=0, transform=None):
        self.transform = transform
        self.demon_dataset = DeMoNDataset(data_dir, input_file, num_right_images=1,
                                          num_left_images=num_left_images, transform=None)
        return

    def __len__(self):
        return len(self.demon_dataset)

    def __getitem__(self, idx):
        sample = self.demon_dataset[idx]

        # Remove outer lists around right image.
        assert(len(sample["right_filename"]) == 1)
        sample["right_filename"] = sample["right_filename"][0]
        sample["right_image"] = sample["right_image"][0]
        sample["right_depthmap_true"] = sample["right_depthmap_true"][0]
        sample["T_right_in_left"] = sample["T_right_in_left"][0]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_groundtruth_disparity(self, image_file):
        idx = self.demon_dataset.left_filename_to_idx[image_file]
        sample = self.__getitem__(idx)
        disparity = depthmap_utils.depthmap_to_disparity(
            sample["K"][0, :3, :3].cpu().numpy(),
            sample["T_right_in_left"][0, ...].cpu().numpy(),
            sample["left_depthmap_true"][0, ...].cpu().numpy())
        return disparity
