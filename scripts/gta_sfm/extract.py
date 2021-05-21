#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import io

import PIL.Image

import numpy as np

import h5py

def extract_hdf5_file(hdf5_file, output_dir):
    hdf5_dict = h5py.File(hdf5_file, "r")
    num_images = len(hdf5_dict.keys()) // 4

    color_dir = os.path.join(output_dir, "color")
    os.makedirs(color_dir)

    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir)

    intrinsics_stream = open(os.path.join(output_dir, "intrinsics.txt"), "w")
    intrinsics_stream.write("# image_id K3x3\n")

    poses_stream = open(os.path.join(output_dir, "poses.txt"), "w")
    poses_stream.write("# image_id pose4x4\n")

    for idx in range(num_images):
        img_key = "image_%d"%idx
        K_key = "K_%d"%idx
        pose_key = "pose_%d"%idx
        depth_key = "depth_%d"%idx

        img = PIL.Image.open(io.BytesIO(hdf5_dict[img_key][:]))
        K = hdf5_dict[K_key][:]
        pose = hdf5_dict[pose_key][:]
        depth = hdf5_dict[depth_key][:]

        image_id_str = "{:06d}".format(idx)

        img.save(os.path.join(color_dir, "{}.jpg".format(image_id_str)))
        np.save(os.path.join(depth_dir, "{}.npy".format(image_id_str)), depth)

        intrinsics_stream.write("{} ".format(image_id_str))
        for row in range(K.shape[0]):
            for col in range(K.shape[1]):
                intrinsics_stream.write("{} ".format(K[row, col]))
        intrinsics_stream.write("\n")

        poses_stream.write("{} ".format(image_id_str))
        for row in range(pose.shape[0]):
            for col in range(pose.shape[1]):
                poses_stream.write("{} ".format(pose[row, col]))
        poses_stream.write("\n")

    return

def main():
    # Extract train data.
    train_files = glob.glob("./train_hdf5/*.hdf5")
    assert(len(train_files) == 200)
    train_output_dir = "./train"
    os.makedirs(train_output_dir)
    for idx in range(len(train_files)):
        train_file = train_files[idx]
        sequence = os.path.splitext(os.path.basename(train_file))[0]
        sequence_dir = os.path.join(train_output_dir, sequence)
        os.makedirs(sequence_dir)
        extract_hdf5_file(train_file, sequence_dir)
        print("Extracted training hdf5 {}/{}".format(idx, len(train_files)))

    # Extract test data.
    test_files = glob.glob("./test_hdf5/*.hdf5")
    assert(len(test_files) == 19)
    test_output_dir = "./test"
    os.makedirs(test_output_dir)
    for idx in range(len(test_files)):
        test_file = test_files[idx]
        sequence = os.path.splitext(os.path.basename(test_file))[0]
        sequence_dir = os.path.join(test_output_dir, sequence)
        os.makedirs(sequence_dir)
        extract_hdf5_file(test_file, sequence_dir)
        print("Extracted test hdf5 {}/{}".format(idx, len(test_files)))

    return

if __name__ == '__main__':
    main()
