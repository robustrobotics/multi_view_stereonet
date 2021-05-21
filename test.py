#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Massachusetts Institute of Technology
#
# @file test.py
# @author W. Nicholas Greene
# @date 2020-10-07 18:18:21 (Wed)

import os
import argparse

import yaml

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import torch

import datasets.multi_view_stereo_dataset as mvsd

import datasets.gta_sfm_dataset as gtad
import datasets.demon_dataset as dd

import stereo.image_predictor as ip

from multi_view_stereonet.multi_view_stereonet import MultiViewStereoNet
from multi_view_stereonet import multi_view_stereonet_utils as snu

from utils import pytorch_utils
from utils import image_utils
from utils import visualization
from utils import image_gallery

BATCH_SIZE = 1


def get_depth_prediction_metrics(depthmap_true, depthmap_est):
    """Compute metrics commonly reported for KITTI depth prediction.

    Assumes no invalid inputs (i.e. mask has already been applied).

    Based on Monodepth.
    """
    thresh = np.maximum((depthmap_true / depthmap_est), (depthmap_est / depthmap_true))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (depthmap_true - depthmap_est) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(depthmap_true) - np.log(depthmap_est)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(depthmap_true - depthmap_est) / depthmap_true)

    sq_rel = np.mean(((depthmap_true - depthmap_est)**2) / depthmap_true)

    metrics = {"abs_rel": abs_rel,
               "sq_rel": sq_rel,
               "rmse": rmse,
               "rmse_log": rmse_log,
               "a1": a1,
               "a2": a2,
               "a3": a3}

    return metrics


def write_images(output_dir, image_idx, idepthmap_est, idepthmap_true):
    """Save colormapped depthmap images for debugging.
    """
    cmap = plt.get_cmap("magma")

    # idepthmaps.
    vmin = 0.0
    vmax = np.max(idepthmap_true)

    debug = np.squeeze(cmap((idepthmap_est - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_est.jpg".format(image_idx)))

    debug = np.squeeze(cmap((idepthmap_true - vmin) / (vmax - vmin)))
    debug_image = Image.fromarray(np.uint8(debug[:, :, :3] * 255))
    debug_image.save(os.path.join(output_dir, "idepthmap_{}_true.jpg".format(image_idx)))

    return

def write_losses_header(output_file, loss_dict):
    """Write header to losses file.
    """
    with open(output_file, "w") as ff:
        ff.write("file loss ")
        for key, value in loss_dict.items():
            if type(value) is list:
                for idx in range(len(value)):
                    ff.write("{}{} ".format(key, idx))
            else:
                ff.write("{} ".format(key))

        ff.write("\n")

    return

def write_losses(output_file, left_file, loss, loss_dict):
    """Write losses.
    """
    with open(output_file, "a") as ff:
        ff.write("{} {} ".format(left_file, loss))
        for key, value in loss_dict.items():
            if type(value) is list:
                for vv in value:
                    ff.write("{} ".format(vv.item()))
            else:
                ff.write("{} ".format(value.item()))

        ff.write("\n")

    return

def write_metrics_header(output_file, metrics_dict):
    """Write metrics header.
    """
    with open(output_file, "w") as ff:
        ff.write("file ")
        for key, value in metrics_dict.items():
            ff.write("{} ".format(key))
        ff.write("\n")
    return

def write_metrics(output_file, input_file, metrics_dict):
    """Write metrics as a line in output file.
    """
    with open(output_file, "a") as ff:
        ff.write("{} ".format(input_file))
        for key, value in metrics_dict.items():
            ff.write("{} ".format(value))
        ff.write("\n")

    return

def compute_avg_metrics(metrics_file):
    """Compute average metrics from metrics file.
    """
    keys = None
    with open(metrics_file, "r") as ff:
        header = ff.readline()
        keys = header.split()
    keys = keys[1:] # Skip filename.

    metrics = np.loadtxt(metrics_file, skiprows=1, usecols=range(1, len(keys) + 1))
    avg_metrics = np.mean(metrics, axis=0)

    avg_metrics_dict = {}
    for idx in range(len(keys)):
        avg_metrics_dict[keys[idx]] = avg_metrics[idx]

    avg_metrics_dict["num_samples"] = metrics.shape[0]

    return avg_metrics_dict

def get_groundtruth_depthmap(split, inputs, left_file):
    if "gta_sfm" in split:
        left_depthmap_true = inputs["left_depthmap_true"]
        baselinehw = inputs["baseline"].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        baselinehw = baselinehw.repeat(1, 1, left_depthmap_true.shape[-2], left_depthmap_true.shape[-1])
        left_depthmap_true *= baselinehw
        left_depthmap_true = left_depthmap_true.squeeze().cpu().numpy()
        min_depth = 0.0
        max_depth = 1e3
    elif "demon" in split:
        left_depthmap_true = inputs["left_depthmap_true"]
        baselinehw = inputs["baseline"].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        baselinehw = baselinehw.repeat(1, 1, left_depthmap_true.shape[-2], left_depthmap_true.shape[-1])
        left_depthmap_true *= baselinehw
        left_depthmap_true = left_depthmap_true.squeeze().cpu().numpy()

        # Limits from DPSNet.
        min_depth = 0.5
        max_depth = 10.0

    return left_depthmap_true, min_depth, max_depth

def test(split, device, stereo_network, loader, save_images, output_dir, params):
    """Test network and compute metrics.
    """
    stereo_network.eval()

    num_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch in loader:
            # Forward pass.
            inputs = snu.multi_view_unpack_batch(batch, device, stereo_network.num_levels)
            outputs = snu.multi_view_forward(stereo_network, inputs, params)
            batch_loss, batch_loss_dict, predictions = snu.compute_losses(inputs, outputs, params)

            assert(np.isnan(batch_loss.item()) == False)

            loss += batch_loss.item()
            num_batches += 1

            print("runtime: {:.2f} ms (batch_size: {})".format(
                outputs["stereo_time_ms"], BATCH_SIZE))

            # Convert idepthmap to depthmap.
            batch_left_idepthmap_est = outputs["left_idepthmap_pyr"][0] / inputs["baseline"]
            batch_left_depthmap_est = outputs["left_idepthmap_pyr"][0] / inputs["baseline"]
            batch_left_depthmap_est[batch_left_depthmap_est > 0] = 1.0 / batch_left_depthmap_est[batch_left_depthmap_est > 0]
            for idx in range(batch_left_depthmap_est.shape[0]):
                # Load groundtruth depthmaps.
                left_file = inputs["left_filename"][idx]
                left_depthmap_true, min_depth, max_depth = get_groundtruth_depthmap(split, inputs, left_file)
                left_idepthmap_true = np.copy(left_depthmap_true)
                left_idepthmap_true[left_idepthmap_true > 0] = 1.0 / left_idepthmap_true[left_idepthmap_true > 0]
                mask = (left_depthmap_true > min_depth) & (left_depthmap_true < max_depth)

                if np.sum(mask) <= 0:
                    print("WARNING: No truth for image: {}".format(left_file))
                    continue

                # Assume output is the same size as ground truth.
                left_idepthmap_est = batch_left_idepthmap_est[idx, :, :, :].unsqueeze(0)
                left_depthmap_est = batch_left_depthmap_est[idx, :, :, :].unsqueeze(0)

                left_idepthmap_est = left_idepthmap_est.cpu().numpy().squeeze()
                left_depthmap_est = left_depthmap_est.cpu().numpy().squeeze()

                # Mask where truth and estimate are valid.
                mask = mask & (left_depthmap_est > min_depth) & (left_depthmap_est < max_depth)

                if save_images:
                    left_dir, file_and_ext = os.path.split(left_file)
                    left_dir = left_dir.replace(loader.dataset.data_dir, "") # Strip dataset prefix.
                    left_output_dir = os.path.join(output_dir, left_dir[1:])
                    image_num = os.path.splitext(file_and_ext)[0]
                    if not os.path.exists(left_output_dir):
                        os.makedirs(left_output_dir)
                    assert(os.path.exists(left_output_dir))
                    write_images(left_output_dir, image_num,
                                 left_idepthmap_est, left_idepthmap_true)
                    left_dir_tokens = left_dir.split(os.path.sep)
                    left_dir_tokens = [token for token in left_dir_tokens if token]
                    image_gallery.create_simple_gallery(os.path.join(output_dir, left_dir_tokens[0]), 2)

                # Save losses.
                loss_file = os.path.join(output_dir, "losses.txt")
                if not os.path.exists(loss_file):
                    write_losses_header(loss_file, batch_loss_dict)
                write_losses(loss_file, left_file, batch_loss.item(), batch_loss_dict)

                # Compute depth metrics and write to file.
                depth_metrics_idx = get_depth_prediction_metrics(
                    left_depthmap_true[mask], left_depthmap_est[mask])
                depth_metrics_file = os.path.join(output_dir, "depth_metrics.txt")
                if not os.path.exists(depth_metrics_file):
                    write_metrics_header(depth_metrics_file, depth_metrics_idx)
                write_metrics(depth_metrics_file, left_file, depth_metrics_idx)

                # Save runtime metrics.
                runtime_metrics_file = os.path.join(output_dir, "runtime_metrics.txt")
                if not os.path.exists(runtime_metrics_file):
                    with open(runtime_metrics_file, "w") as stream:
                        stream.write("file runtime_ms\n")
                with open(runtime_metrics_file, "a") as stream:
                    stream.write("{} {}\n".format(left_file, outputs["stereo_time_ms"]))

                print("image: {}, LOSS: {:.2f}, ABS_REL: {:.2f}, A1: {:.2f}, A3: {:.2f}, A3: {:.2f}".format(
                    left_file, batch_loss.item(), depth_metrics_idx["abs_rel"], depth_metrics_idx["a1"],
                    depth_metrics_idx["a2"], depth_metrics_idx["a3"]))

            print("Processed batch {}/{}".format(num_batches, len(loader)))

    loss /= num_batches

    return loss, num_batches

def load_data(data_dir, test_file, params):
    """Load  dataset.
    """
    roll_right_image_180 = False
    add_translation_noise = False
    add_rotation_noise = False
    testing_transforms = mvsd.get_testing_transforms(
        params, roll_right_image_180, add_translation_noise, add_rotation_noise)
    if "gta_sfm" in test_file:
        dataset = gtad.GTASfMMultiViewStereoDataset(
            data_dir, test_file, 0, testing_transforms,
            load_groundtruth_depthmaps=True)
    elif "demon" in test_file:
        dataset = dd.DeMoNDataset(data_dir, test_file,
                                  num_right_images=1, num_left_images=0,
                                  transform=testing_transforms)
    else:
        assert(False)

    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, pin_memory=False)

    return loader

def load_models(device, weights_dir, params):
    # TorchScript has issues for older pytorch versions.
    assert(torch.__version__ >= "1.3.0" )

    stereo_network = torch.jit.load(os.path.join(weights_dir, "stereo_network.pt"))
    stereo_network = stereo_network.to(device)

    stereo_network.eval()

    return stereo_network

def main():
    """Tests loading trained MultiViewStereoNet model and performing inference on multi-view data.
    """
    # Parse args.
    parser = argparse.ArgumentParser(description="Run MultiViewStereoNet inference.")
    parser.add_argument("weights_dir", help="Path to saved model directory.")
    parser.add_argument("data_dir", help="Path to input data.")
    parser.add_argument("test_split", help="Test split file.")
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()

    args.weights_dir = os.path.abspath(args.weights_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    args.test_split = os.path.abspath(args.test_split)

    assert(os.path.exists(args.weights_dir))
    assert(os.path.exists(args.data_dir))
    assert(os.path.exists(args.test_split))

    # Load params.
    params_file = os.path.join(args.weights_dir, "..", "..", "params.yaml")
    assert(os.path.exists(params_file))
    params = yaml.load(open(params_file, "r"), Loader=yaml.FullLoader)

    # params["num_idepth_samples"] = 12
    # params["cost_volume_filter"] = True
    # params["refiners"] = [True, True, True, True, True]

    # Set device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA DEVICE FOUND!")
    else:
        device = torch.device("cpu")
        print("DEFAULTING TO CPU!")

    # Load data.
    loader = load_data(args.data_dir, args.test_split, params)

    # Load models.
    stereo_network = load_models(device, args.weights_dir, params)

    # Create output dir.
    output_dir = "output"
    assert(not os.path.exists(output_dir))
    os.makedirs(output_dir)

    # Evaluate network on test data.
    loss, num_batches= test(
        args.test_split, device, stereo_network, loader,
        args.save_images, output_dir, params)

    # Compute metrics averaged across entire test set.
    avg_losses = compute_avg_metrics(os.path.join(output_dir, "losses.txt"))
    with open(os.path.join(output_dir, "avg_losses.txt"), "w") as ff:
        for key, value in avg_losses.items():
            ff.write("{}: {}\n".format(key, value))

    avg_depth_metrics = compute_avg_metrics(os.path.join(output_dir, "depth_metrics.txt"))
    with open(os.path.join(output_dir, "avg_depth_metrics.txt"), "w") as ff:
        for key, value in avg_depth_metrics.items():
            ff.write("{}: {}\n".format(key, value))

    runtimes = np.loadtxt(os.path.join(output_dir, "runtime_metrics.txt"),
                          skiprows=1, usecols=1)
    mean_runtime = np.mean(runtimes)
    with open(os.path.join(output_dir, "avg_runtime_metrics.txt"), "w") as ff:
        ff.write("runtime_ms: {}\n".format(mean_runtime))
        ff.write("num_samples: {}\n".format(len(runtimes)))

    if "demon" in args.test_split:
        # Compute average metrics per scene type in demon.
        demon_types = ["mvs", "sun3d", "rgbd", "scenes11"]
        lines = []
        with open(os.path.join(output_dir, "depth_metrics.txt"), "r") as ff:
            lines = ff.readlines()

        header = lines[0]
        for demon_type in demon_types:
            metric_lines = [line for line in lines if demon_type in line]

            with open(os.path.join(output_dir, "depth_metrics_{}.txt".format(demon_type)), "w") as ff:
                ff.write(header)
                for line in metric_lines:
                    ff.write(line)

            avg_demon_metrics = compute_avg_metrics(os.path.join(output_dir, "depth_metrics_{}.txt".format(demon_type)))
            with open(os.path.join(output_dir, "avg_depth_metrics_{}.txt".format(demon_type)), "w") as ff:
                for key, value in avg_demon_metrics.items():
                    ff.write("{}: {}\n".format(key, value))

    return

if __name__ == '__main__':
    main()
