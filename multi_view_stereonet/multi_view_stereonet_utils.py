# Copyright 2021 Massachusetts Institute of Technology
#
# @file multi_view_stereonet_utils.py
# @author W. Nicholas Greene
# @date 2020-02-19 13:53:42 (Wed)

import os
import hashlib

import PIL

import torch
import torchvision as tv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from . import losses

from stereo import image_predictor as ip

from utils import losses as ulosses
from utils import pytorch_utils
from utils import image_utils
from utils import visualization
from utils import image_gallery

def log_losses(epoch, batch, step, loss, loss_dict, output_file):
    """Log losses to disk.
    """
    if not os.path.exists(output_file):
        with open(output_file, "w") as ff:
            ff.write("epoch batch step loss ")
            for key, value in loss_dict.items():
                if type(value) is list:
                    for idx in range(len(value)):
                        ff.write("{}{} ".format(key, idx))
                else:
                    ff.write("{} ".format(key))

            ff.write("\n")

    with open(output_file, "a") as ff:
        ff.write("{} {} {} {} ".format(epoch, batch, step, loss))
        for key, value in loss_dict.items():
            if type(value) is list:
                for vv in value:
                    ff.write("{} ".format(vv.item()))
            else:
                ff.write("{} ".format(value.item()))

        ff.write("\n")

    return

def log_validation_metrics(epoch, loss, metrics, output_file):
    """Log validation metrics to disk.
    """
    if not os.path.exists(output_file):
        with open(output_file, "w") as target:
            target.write("epoch loss ")
            for key, value in metrics.items():
                target.write("{} ".format(key))
            target.write("\n")

    with open(output_file, "a") as target:
        target.write("{} {} ".format(epoch, loss))
        for key, value in metrics.items():
            target.write("{} ".format(value))
        target.write("\n")

    return

def plot_losses(loss_file, output_dir, smooth=True):
    """ Make plots of all the different losses.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read in losses from file.
    loss_keys = []
    with open(loss_file, "r") as target:
        loss_keys = target.readline()
    loss_keys = loss_keys.split()[3:] # First three are epochs, batch, and steps.

    epochs = np.loadtxt(loss_file, skiprows=1, usecols=0, ndmin=1)
    batch = np.loadtxt(loss_file, skiprows=1, usecols=1, ndmin=1)
    steps = np.loadtxt(loss_file, skiprows=1, usecols=2, ndmin=1)
    losses = np.loadtxt(loss_file, skiprows=1, usecols=range(3, len(loss_keys) + 3), ndmin=2)
    assert(len(loss_keys) == losses.shape[-1])

    if np.max(epochs) == 0:
        # Use steps for x axis.
        xaxis = steps
        xlabel = "Steps"
    else:
        # Create fractional epoch for x axis.
        batch_per_epoch = np.max(batch)
        xaxis = epochs + batch / batch_per_epoch
        xlabel = "Epoch"

    for idx in range(len(loss_keys)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        loss = losses[:, idx]
        final_loss = loss[-1]
        slope = 0.0
        if len(xaxis) > 2 and smooth:
            # Smooth signal so that overall training trend isn't diluted.
            max_smoothed_samples = 100
            smoothing_factor = np.int(np.ceil(float(len(xaxis)) / max_smoothed_samples))
            xidxs = np.arange(len(xaxis))
            bin_edges = xidxs[::smoothing_factor]
            bin_counts = np.diff(bin_edges)

            # Compute running mean/standard deviation using integral
            # images/summed area tables. Assumes np.cumsum is inclusive.
            # See here: https://en.wikipedia.org/wiki/Summed-area_table
            running_sum = np.cumsum(loss) - loss # Exclusive cumsum
            running_sum2 = np.cumsum(loss**2) - loss**2 # Exclusive cumsum
            S1 = running_sum[bin_edges[1:]] - running_sum[bin_edges[:-1]]
            S2 = running_sum2[bin_edges[1:]] - running_sum2[bin_edges[:-1]]
            mean = S1 / bin_counts
            var = S2 / bin_counts - S1**2 / bin_counts**2 + 1e-8
            std = np.sqrt(var)

            ax.plot(xaxis[bin_edges[1:]], mean, "b")
            ax.plot(xaxis[bin_edges[1:]], mean + std, c="0.5", linestyle="--")
            ax.plot(xaxis[bin_edges[1:]], mean - std, c="0.5", linestyle="--")
            final_loss = mean[-1]
            slope = (mean[-1] - mean[len(mean) // 2]) / (xaxis[bin_edges[1:]][-1] - xaxis[bin_edges[1:]][len(mean) // 2])
            ymin = mean[-1] - std[-1]
            ymin -= 0.1 * ymin
            ymax = np.percentile(mean, 90)
            ymax += 0.1 * ymax
        else:
            ax.plot(xaxis, loss, "b")
            ymin = np.min(loss)
            ymax = np.percentile(loss, 90)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("{}".format(loss_keys[idx]))
        ax.set_title("{}: {:.3f}, slope: {:.3f}".format(loss_keys[idx], final_loss, slope))

        ax.grid(True)
        ax.axis("tight")
        ax.set_ylim(ymin, ymax)

        fig.savefig(os.path.join(output_dir, "{}.jpg".format(loss_keys[idx])))
        fig.savefig(os.path.join(output_dir, "{}.pdf".format(loss_keys[idx])))

        plt.close(fig)

    image_gallery.create_simple_gallery(output_dir)

    return

def plot_validation(training_file, validation_file, output_dir, smooth=True):
    """Make plots comparing training performance to validation performance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_keys = []
    with open(training_file, "r") as target:
        training_keys = target.readline()
    training_keys = training_keys.split()

    validation_keys = []
    with open(validation_file, "r") as target:
        validation_keys = target.readline()
    validation_keys = validation_keys.split()

    training_data = np.loadtxt(training_file, skiprows=1, ndmin=2)
    validation_data = np.loadtxt(validation_file, skiprows=1, ndmin=2)

    # Plot training vs. validation loss.
    fig_tv_loss = plt.figure()
    ax_tv_loss = fig_tv_loss.add_subplot(111)

    training_epoch_idx = 0
    training_batch_idx = 1
    training_loss_idx = 3
    assert(training_keys[training_epoch_idx] == "epoch")
    assert(training_keys[training_batch_idx] == "batch")
    assert(training_keys[training_loss_idx] == "loss")
    batch_per_epoch = np.max(training_data[:, training_batch_idx])
    training_xaxis = training_data[:, training_epoch_idx] + training_data[:, training_batch_idx] / batch_per_epoch

    training_loss = training_data[:, training_loss_idx]
    final_train_loss = training_loss[-1]
    if len(training_loss) > 2 and smooth:
        # Smooth signal so that overall training trend isn't diluted.
        max_smoothed_samples = 100
        smoothing_factor = np.int(np.ceil(float(len(training_xaxis)) / max_smoothed_samples))
        xidxs = np.arange(len(training_xaxis))
        bin_edges = xidxs[::smoothing_factor]
        bin_counts = np.diff(bin_edges)

        # Compute running mean/standard deviation using integral
        # images/summed area tables. Assumes np.cumsum is inclusive.
        # See here: https://en.wikipedia.org/wiki/Summed-area_table
        running_sum = np.cumsum(training_loss) - training_loss # Exclusive cumsum
        running_sum2 = np.cumsum(training_loss**2) - training_loss**2 # Exclusive cumsum
        S1 = running_sum[bin_edges[1:]] - running_sum[bin_edges[:-1]]
        S2 = running_sum2[bin_edges[1:]] - running_sum2[bin_edges[:-1]]
        mean = S1 / bin_counts
        var = S2 / bin_counts - S1**2 / bin_counts**2 + 1e-8
        std = np.sqrt(var)

        final_train_loss = mean[-1]
        ax_tv_loss.plot(training_xaxis[bin_edges[1:]], mean, "b", label="train")
        # ax_tv_loss.plot(training_xaxis[bin_edges[1:]], mean + std, c="0.5", linestyle="--")
        # ax_tv_loss.plot(training_xaxis[bin_edges[1:]], mean - std, c="0.5", linestyle="--")
    else:
        ax_tv_loss.plot(training_xaxis, training_loss, "b", label="train")

    val_epoch_idx = 0
    val_loss_idx = 1
    assert(validation_keys[val_epoch_idx] == "epoch")
    assert(validation_keys[val_loss_idx] == "loss")
    ax_tv_loss.plot(validation_data[:, val_epoch_idx] + 1, validation_data[:, val_loss_idx], "r", label="val")
    final_val_loss = validation_data[-1, val_loss_idx]

    ax_tv_loss.set_xlabel("Epoch")
    ax_tv_loss.set_ylabel("Loss")
    ax_tv_loss.set_title("Training ({:.3f}) vs. Validation Loss ({:.3f})".format(final_train_loss, final_val_loss))

    ax_tv_loss.grid(True)
    ax_tv_loss.axis("tight")

    plt.legend(loc="best")

    fig_tv_loss.savefig(os.path.join(output_dir, "training_validation_loss.jpg"))
    fig_tv_loss.savefig(os.path.join(output_dir, "training_validation_loss.pdf"))

    plt.close(fig_tv_loss)

    image_gallery.create_simple_gallery(output_dir)

    return

def log_debug_idepthmap(epoch, step, image_id, left, right, truth, idepthmap, output_dir):
    """Log a idepthmap debug image to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save input images.
    tv.utils.save_image((left+1)*0.5, os.path.join(output_dir, "{:d}_left_input.jpg".format(image_id)))
    tv.utils.save_image((right+1)*0.5, os.path.join(output_dir, "{:d}_right_input.jpg".format(image_id)))

    magma = plt.get_cmap("magma")

    # Save truth.
    max_idepth = None
    if truth is not None:
        max_idepth = torch.max(truth)
        idepthmap_true_debug = visualization.apply_cmap(truth, 0.0, max_idepth, magma)
        plt.imsave(os.path.join(output_dir, "{:d}_left_ground_truth.jpg".format(image_id)),
                   idepthmap_true_debug.squeeze())

    # Save estimated idepthmap.
    idepthmap_debug = visualization.apply_cmap(idepthmap, 0.0, max_idepth, magma)
    idepthmap_filename = "{:d}_{:04d}.jpg".format(image_id, epoch)
    plt.imsave(os.path.join(output_dir, idepthmap_filename), idepthmap_debug.squeeze())

    return

def log_debug_occlusion_mask(epoch, step, image_id, mask, truth, output_dir):
    """Log occlusion mask debug image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask_debug = mask.detach().squeeze().cpu().numpy().astype(np.uint8) * 255
    mask_debug_filename = "{:d}_{:04d}.jpg".format(image_id, epoch)
    plt.imsave(os.path.join(output_dir, mask_debug_filename),
               mask_debug, cmap=matplotlib.cm.get_cmap("gray"))

    if truth is not None:
        mask_true = truth.detach().squeeze().cpu().numpy().astype(np.uint8) * 255
        mask_true_filename = "{:d}_true.jpg".format(image_id)
        plt.imsave(os.path.join(output_dir, mask_true_filename),
                   mask_true, cmap=matplotlib.cm.get_cmap("gray"))

    return

def log_debug_images(epoch, step, batch_idx, inputs, outputs, predictions, output_dir, params):
    """Log debug images to disk.

    Will only log the <batch_idx> image in the batch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Some datasets don't have unique IDs for each image. Hash the image
    # filename to get one to ease logging and save the mapping.
    left_filename = inputs["left_filename"][batch_idx]
    right_filename = inputs["right_filename"][batch_idx]
    left_id = int(hashlib.sha1(left_filename.encode("utf-8")).hexdigest(), 16) % 1000000000
    right_id = int(hashlib.sha1(right_filename.encode("utf-8")).hexdigest(), 16) % 1000000000
    ids_file = os.path.join(output_dir, "image_ids.txt")
    if not os.path.exists(ids_file):
        with open(ids_file, "w") as ff:
            ff.write("left_id left_filename right_id right_filename\n")
            ff.write("{} {} {} {}\n".format(left_id, left_filename, right_id, right_filename))
    else:
        # Check if this image is already logged.
        left_ids = []
        with open(ids_file, "r") as ff:
            lines = ff.readlines()
            lines = lines[1:]
            for line in lines:
                tokens = line.split()
                left_ids.append(int(tokens[0]))

        left_ids_set = set(left_ids)
        if left_id not in left_ids_set:
            with open(ids_file, "a") as ff:
                ff.write("{} {} {} {}\n".format(left_id, left_filename, right_id, right_filename))

    # Save idepthmaps.
    # NOTE: Right images will be saved using left_id to maintain the same order when viewed.
    for lvl in range(len(outputs["left_idepthmap_pyr"])):
        if outputs["left_idepthmap_pyr"][lvl] is None:
            continue
        left_idepthmap_dir = os.path.join(output_dir, "left_idepthmap{}".format(lvl))
        log_debug_idepthmap(epoch, step, left_id,
                            inputs["left_image_pyr"][0][batch_idx, ...],
                            inputs["right_image_pyr"][0][batch_idx, ...],
                            inputs["left_idepthmap_true"][batch_idx, ...].unsqueeze(0),
                            outputs["left_idepthmap_pyr"][lvl][batch_idx, ...].unsqueeze(0),
                            left_idepthmap_dir)
        image_gallery.create_training_gallery(left_idepthmap_dir)

    # Save raw idepthmap at coarsest level.
    left_idepthmap_raw_dir = os.path.join(output_dir, "left_idepthmap_raw{}".format(len(outputs["left_idepthmap_raw_pyr"])-1))
    log_debug_idepthmap(epoch, step, left_id,
                        inputs["left_image_pyr"][0][batch_idx, ...],
                        inputs["right_image_pyr"][0][batch_idx, ...],
                        inputs["left_idepthmap_true"][batch_idx, ...].unsqueeze(0),
                        outputs["left_idepthmap_raw_pyr"][-1][batch_idx, ...].unsqueeze(0),
                        left_idepthmap_raw_dir)
    image_gallery.create_training_gallery(left_idepthmap_raw_dir)

    if "right_idepthmap_pyr" in outputs:
        right_idepthmap0_dir = os.path.join(output_dir, "right_idepthmap0")
        log_debug_idepthmap(epoch, step, left_id,
                            inputs["right_image_pyr"][0][batch_idx, ...],
                            inputs["left_image_pyr"][0][batch_idx, ...],
                            inputs["right_idepthmap_true"][batch_idx, ...].unsqueeze(0),
                            outputs["right_idepthmap_pyr"][0][batch_idx, ...].unsqueeze(0),
                            os.path.join(output_dir, "right_idepthmap0"))
        image_gallery.create_training_gallery(right_idepthmap0_dir)

    # Save occlusion masks.
    # NOTE: Right images will be saved using left_id to maintain the same order when viewed.
    if "left_occlusion_mask_pyr" in predictions:
        left_occlusion_mask0_dir = os.path.join(output_dir, "left_occlusion_mask0")
        log_debug_occlusion_mask(epoch, step, left_id,
                                 predictions["left_occlusion_mask_pyr"][0][batch_idx, ...],
                                 predictions["left_occlusion_mask_true"][batch_idx, ...],
                                 left_occlusion_mask0_dir)
        image_gallery.create_training_gallery(left_occlusion_mask0_dir)

    if "right_occlusion_mask_pyr" in predictions:
        right_occlusion_mask0_dir = os.path.join(output_dir, "right_occlusion_mask0")
        log_debug_occlusion_mask(epoch, step, left_id,
                                 predictions["right_occlusion_mask_pyr"][0][batch_idx, ...],
                                 predictions["right_occlusion_mask_true"][batch_idx, ...],
                                 right_occlusion_mask0_dir)
        image_gallery.create_training_gallery(right_occlusion_mask0_dir)

    # Save warped images.
    if "right_image_warped" in outputs:
        right_image_warped_dir = os.path.join(output_dir, "right_image_warped")
        if not os.path.exists(right_image_warped_dir):
            os.makedirs(right_image_warped_dir)
        tv.utils.save_image(inputs["left_image_pyr"][0][batch_idx, ...],
                            os.path.join(right_image_warped_dir, "{:d}_left_input.jpg".format(left_id)))
        tv.utils.save_image(inputs["right_image_pyr"][0][batch_idx, ...],
                            os.path.join(right_image_warped_dir, "{:d}_right_input.jpg".format(left_id)))
        tv.utils.save_image(outputs["right_image_warped"][0][batch_idx, :, 0, :, :],
                            os.path.join(right_image_warped_dir, "{:d}_right_warped_max_depth.jpg".format(left_id)))
        tv.utils.save_image(outputs["right_image_warped"][0][batch_idx, :, -1, :, :],
                            os.path.join(right_image_warped_dir, "{:d}_right_warped_min_depth.jpg".format(left_id)))
        image_gallery.create_simple_gallery(right_image_warped_dir, 4)

    if "right_feature_volume" in outputs:
        right_feature_volume_dir = os.path.join(output_dir, "right_feature_volume")
        if not os.path.exists(right_feature_volume_dir):
            os.makedirs(right_feature_volume_dir)
        tv.utils.save_image(outputs["left_feature_pyr"][-1][batch_idx, :3, :, :],
                            os.path.join(right_feature_volume_dir, "{:d}_left_features.jpg".format(left_id)))
        tv.utils.save_image(outputs["right_feature_volume"][-1][batch_idx, :3, 0, :, :],
                            os.path.join(right_feature_volume_dir, "{:d}_right_features_max_depth.jpg".format(left_id)))
        tv.utils.save_image(outputs["right_feature_volume"][-1][batch_idx, :3, -1, :, :],
                            os.path.join(right_feature_volume_dir, "{:d}_right_features_min_depth.jpg".format(left_id)))
        image_gallery.create_simple_gallery(right_feature_volume_dir, 3)

    return

def unpack_batch(batch, device, num_levels):
    """Unpack a batch of data.
    """
    left_image, right_image = batch["left_image"], batch["right_image"]
    left_image = left_image.to(device)
    right_image = right_image.to(device)

    K = batch["K"].to(device)
    T_right_in_left = batch["T_right_in_left"].to(device)

    # Remove unnecessary channel dimension.
    K = torch.squeeze(K, dim=1)
    T_right_in_left = torch.squeeze(T_right_in_left, dim=1)

    # Normalize baseline to be 1m.
    baseline = torch.sqrt(torch.sum(T_right_in_left[:, :3, 3]**2, 1))
    baseline_mask = baseline > 0
    assert(torch.sum(baseline_mask) == baseline.shape[0])

    baseline3 = torch.unsqueeze(baseline, dim=1)
    baseline3 = baseline3.repeat(1, 3)
    T_right_in_left[:, 0:3, 3] /= baseline3

    T_left_in_right = torch.inverse(T_right_in_left)

    # Create image pyramids.
    left_image_pyr = image_utils.build_image_pyramid(left_image, num_levels)
    right_image_pyr = image_utils.build_image_pyramid(right_image, num_levels)

    K_pyr = [K]
    for idx in range(1, num_levels):
        # NOTE: Principal point should lie in the same place relative to the
        # image plane after the transformation, which is tricky depending on
        # your pixel coordinate convention. Suppose we resize the image by a
        # factor s. A pixel (x, y) in the old image is mapped to the following
        # coordinates in the resized image (assuming we use the convention that
        # the origin lies in the center of the top-left pixel):
        #
        # x' = s * (x + 0.5) - 0.5
        # y' = s * (y + 0.5) - 0.5
        #
        # The new intrinsic matrix is then given by:
        #
        # K_new = S * K_old
        #
        # where S = [[s, 0, s*0.5 - 0.5], [0, s, s*0.5 - 0.5], [0, 0, 1]]
        #
        # See here for a more info:
        # https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        x_factor = float(left_image_pyr[idx].shape[-1]) / (left_image_pyr[0].shape[-1])
        y_factor = float(left_image_pyr[idx].shape[-2]) / (left_image_pyr[0].shape[-2])
        K_lvl = torch.clone(K)
        K_lvl[:, 0, 0] *= x_factor
        K_lvl[:, 1, 1] *= y_factor
        K_lvl[:, 0, 2] = x_factor * (K_lvl[:, 0, 2] + 0.5) - 0.5
        K_lvl[:, 1, 2] = y_factor * (K_lvl[:, 1, 2] + 0.5) - 0.5
        K_pyr.append(K_lvl)

    inputs = {"left_filename": batch["left_filename"],
              "right_filename": batch["right_filename"],
              "T_right_in_left": T_right_in_left,
              "T_left_in_right": T_left_in_right,
              "K_pyr": K_pyr,
              "left_image_pyr": left_image_pyr,
              "right_image_pyr": right_image_pyr,
              "baseline": baseline}

    if "left_disparity_true" in batch:
        left_disparity_true, right_disparity_true = batch["left_disparity_true"], batch["right_disparity_true"]
        left_disparity_true = left_disparity_true.to(device)
        right_disparity_true = right_disparity_true.to(device)
        inputs["left_disparity_true"] = left_disparity_true
        inputs["right_disparity_true"] = right_disparity_true

    if "left_depthmap_true" in batch:
        left_depthmap_true, right_depthmap_true = batch["left_depthmap_true"], batch["right_depthmap_true"]
        left_depthmap_true = left_depthmap_true.to(device)
        right_depthmap_true = right_depthmap_true.to(device)

        baselinehw = baseline.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        baselinehw = baselinehw.repeat(1, 1, left_depthmap_true.shape[-2], left_depthmap_true.shape[-1])

        inputs["left_depthmap_true"] = left_depthmap_true / baselinehw
        inputs["right_depthmap_true"] = right_depthmap_true / baselinehw

        inputs["left_idepthmap_true"] = inputs["left_depthmap_true"].clone()
        left_mask = inputs["left_idepthmap_true"] > 0
        inputs["left_idepthmap_true"][left_mask] = 1.0 / inputs["left_idepthmap_true"][left_mask]

        inputs["right_idepthmap_true"] = inputs["right_depthmap_true"].clone()
        right_mask = inputs["right_idepthmap_true"] > 0
        inputs["right_idepthmap_true"][right_mask] = 1.0 / inputs["right_idepthmap_true"][right_mask]

    assert(inputs["left_image_pyr"][0].dtype == torch.float32)

    return inputs

def forward(stereo_network, inputs, params):
    """Perform a forward pass to create outputs.
    """
    left_stereo_tick, left_stereo_tock = pytorch_utils.start_timer()
    left_outputs = stereo_network(
        inputs["left_image_pyr"],
        inputs["K_pyr"],
        [inputs["T_right_in_left"]],
        [inputs["right_image_pyr"]],
        params["num_idepth_samples"],
        params["cost_volume_filter"],
        params["refiners"])
    left_stereo_time_ms = pytorch_utils.stop_timer(left_stereo_tick, left_stereo_tock)

    outputs = {"left_idepthmap_pyr": left_outputs["left_idepthmap_pyr"],
               "left_idepthmap_raw_pyr": left_outputs["left_idepthmap_raw_pyr"],
               "left_idepthmap_mask_pyr": left_outputs["left_idepthmap_mask_pyr"],
               "stereo_time_ms": left_stereo_time_ms}

    if params["estimate_right_idepthmap"]:
        right_stereo_tick, right_stereo_tock = pytorch_utils.start_timer()
        right_outputs = stereo_network(
            inputs["right_image_pyr"],
            inputs["K_pyr"],
            [inputs["T_left_in_right"]],
            [inputs["left_image_pyr"]],
            params["num_idepth_samples"],
            params["cost_volume_filter"],
            params["refiners"])
        right_stereo_time_ms = pytorch_utils.stop_timer(right_stereo_tick, right_stereo_tock)
    
        outputs["right_idepthmap_pyr"] = right_outputs["left_idepthmap_pyr"]
        outputs["right_idepthmap_raw_pyr"] = right_outputs["left_idepthmap_raw_pyr"]
        outputs["right_idepthmap_mask_pyr"] = right_outputs["left_idepthmap_mask_pyr"]
        outputs["stereo_time_ms"] = 0.5 * (outputs["stereo_time_ms"] + right_stereo_time_ms)

    return outputs

def multi_view_unpack_batch(batch, device, num_levels):
    """Unpack a batch of data.
    """
    left_image = batch["left_image"]
    left_image = left_image.to(device)

    right_image = batch["right_image"]
    for idx in range(len(right_image)):
        right_image[idx] = right_image[idx].to(device)

    left_image_pyr = image_utils.build_image_pyramid(left_image, num_levels)

    K = batch["K"].to(device)
    K = torch.squeeze(K, dim=1)
    K_pyr = [K]
    for idx in range(1, num_levels):
        # NOTE: Principal point should lie in the same place relative to the
        # image plane after the transformation, which is tricky depending on
        # your pixel coordinate convention. Suppose we resize the image by a
        # factor s. A pixel (x, y) in the old image is mapped to the following
        # coordinates in the resized image (assuming we use the convention that
        # the origin lies in the center of the top-left pixel):
        #
        # x' = s * (x + 0.5) - 0.5
        # y' = s * (y + 0.5) - 0.5
        #
        # The new intrinsic matrix is then given by:
        #
        # K_new = S * K_old
        #
        # where S = [[s, 0, s*0.5 - 0.5], [0, s, s*0.5 - 0.5], [0, 0, 1]]
        #
        # See here for a more info:
        # https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        x_factor = float(left_image_pyr[idx].shape[-1]) / (left_image_pyr[0].shape[-1])
        y_factor = float(left_image_pyr[idx].shape[-2]) / (left_image_pyr[0].shape[-2])
        K_lvl = torch.clone(K)
        K_lvl[:, 0, 0] *= x_factor
        K_lvl[:, 1, 1] *= y_factor
        K_lvl[:, 0, 2] = x_factor * (K_lvl[:, 0, 2] + 0.5) - 0.5
        K_lvl[:, 1, 2] = y_factor * (K_lvl[:, 1, 2] + 0.5) - 0.5
        K_pyr.append(K_lvl)

    T_right_in_left = []
    T_left_in_right = []
    right_image_pyr = []
    for right_idx in range(len(batch["T_right_in_left"])):
        T_r_in_l = batch["T_right_in_left"][right_idx].to(device)
        T_r_in_l = torch.squeeze(T_r_in_l, dim=1)
        T_l_in_r = torch.inverse(T_r_in_l)

        T_right_in_left.append(T_r_in_l)
        T_left_in_right.append(T_l_in_r)
        right_image_pyr.append(image_utils.build_image_pyramid(right_image[right_idx], num_levels))

    # Normalize poses by the baseline to the first right camera.
    baseline = torch.sqrt(torch.sum(T_right_in_left[0][:, :3, 3]**2, 1))
    baseline_mask = baseline > 0
    assert(torch.sum(baseline_mask) == baseline.shape[0])
    baseline3 = torch.unsqueeze(baseline, dim=1)
    baseline3 = baseline3.repeat(1, 3)
    for right_idx in range(len(batch["T_right_in_left"])):
        T_right_in_left[right_idx][:, 0:3, 3] /= baseline3
        T_left_in_right[right_idx][:, 0:3, 3] /= baseline3

    inputs = {"left_filename": batch["left_filename"],
              "right_filename": batch["right_filename"],
              "T_right_in_left": T_right_in_left,
              "T_left_in_right": T_left_in_right,
              "K_pyr": K_pyr,
              "left_image_pyr": left_image_pyr,
              "right_image_pyr": right_image_pyr,
              "baseline": baseline}

    if "left_depthmap_true" in batch:
        left_depthmap_true = batch["left_depthmap_true"]
        left_depthmap_true = left_depthmap_true.to(device)

        baselinehw = baseline.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        baselinehw = baselinehw.repeat(1, 1, left_depthmap_true.shape[-2], left_depthmap_true.shape[-1])
        inputs["left_depthmap_true"] = left_depthmap_true / baselinehw

        inputs["left_idepthmap_true"] = inputs["left_depthmap_true"].clone()
        left_mask = inputs["left_idepthmap_true"] > 0
        inputs["left_idepthmap_true"][left_mask] = 1.0 / inputs["left_idepthmap_true"][left_mask]

        right_depthmap_true = batch["right_depthmap_true"]
        for right_idx in range(len(batch["right_depthmap_true"])):
            right_depthmap_true[right_idx] = right_depthmap_true[right_idx].to(device)
            right_depthmap_true[right_idx] /= baselinehw
        inputs["right_depthmap_true"] = right_depthmap_true

        right_idepthmap_true = [right.clone() for right in inputs["right_depthmap_true"]]
        for right_idx in range(len(batch["right_depthmap_true"])):
            right_mask = right_idepthmap_true[right_idx] > 0
            right_idepthmap_true[right_idx][right_mask] = 1.0 / right_idepthmap_true[right_idx][right_mask]
        inputs["right_idepthmap_true"] = right_idepthmap_true

    assert(inputs["left_image_pyr"][0].dtype == torch.float32)

    return inputs

def multi_view_forward(stereo_network, inputs, params):
    """Perform a forward pass to create outputs.
    """
    left_stereo_tick, left_stereo_tock = pytorch_utils.start_timer()
    left_outputs = stereo_network(
        inputs["left_image_pyr"],
        inputs["K_pyr"],
        inputs["T_right_in_left"],
        inputs["right_image_pyr"],
        params["num_idepth_samples"],
        params["cost_volume_filter"],
        params["refiners"])
    left_stereo_time_ms = pytorch_utils.stop_timer(left_stereo_tick, left_stereo_tock)

    outputs = {"left_idepthmap_pyr": left_outputs["left_idepthmap_pyr"],
               "left_idepthmap_raw_pyr": left_outputs["left_idepthmap_raw_pyr"],
               "left_idepthmap_mask_pyr": left_outputs["left_idepthmap_mask_pyr"],
               "stereo_time_ms": left_stereo_time_ms}

    return outputs

def compute_losses(inputs, outputs, params):
    """Compute losses.
    """
    loss = 0.0
    loss_dict = {}
    predictions = {}

    if params["supervision_factor"] > 0.0:
        # Compute supervised loss ----------------------------------------------
        truth_size = inputs["left_idepthmap_true"].shape[-2:]
        idepth_scale_factor = 100.0 # torch.mean(inputs["K_pyr"][0][:, 0, 0])

        supervised_losses = []

        # Loss from left idepthmap.
        # left_true_mask = (inputs["left_idepthmap_true"] >= 0) & (inputs["left_idepthmap_true"] < max_idepthmap)
        left_true_mask = inputs["left_idepthmap_true"] > 0
        assert(torch.sum(left_true_mask) > 0)
        for lvl in range(len(outputs["left_idepthmap_pyr"])):
            if outputs["left_idepthmap_pyr"][lvl] is None:
                continue
            supervised_losses.append(losses.supervised_idepthmap_loss(
                outputs["left_idepthmap_pyr"][lvl], inputs["left_idepthmap_true"],
                left_true_mask, idepth_scale_factor))

        # Loss from raw base level.
        supervised_losses.append(losses.supervised_idepthmap_loss(
            outputs["left_idepthmap_raw_pyr"][-1], inputs["left_idepthmap_true"],
            left_true_mask, idepth_scale_factor))

        if "right_idepthmap_pyr" in outputs:
            # Loss from right idepthmap.
            # right_true_mask = (inputs["right_idepthmap_true"] >= 0) & (inputs["right_idepthmap_true"] < max_idepthmap)
            right_true_mask = inputs["right_idepthmap_true"] > 0
            assert(torch.sum(right_true_mask) > 0)
            for lvl in range(len(outputs["right_idepthmap_pyr"])):
                if outputs["right_idepthmap_pyr"][lvl] is None:
                    continue
                supervised_losses.append(losses.supervised_idepthmap_loss(
                    outputs["right_idepthmap_pyr"][lvl], inputs["right_idepthmap_true"],
                    right_true_mask, idepth_scale_factor))

        loss_dict["supervised_losses"] = supervised_losses

        supervised_loss = sum(supervised_losses) / len(supervised_losses)
        loss += params["supervision_factor"] * supervised_loss
        loss_dict["supervised_loss"] = supervised_loss

    if "right_idepthmap_pyr" in outputs:
        # Compute occlusion masks---------------------------------------------------
        left_occlusion_mask_pyr = [None for _ in range(len(outputs["left_idepthmap_pyr"]))]
        right_occlusion_mask_pyr = [None for _ in range(len(outputs["right_idepthmap_pyr"]))]
        for lvl in range(len(outputs["left_idepthmap_pyr"])):
            if outputs["left_idepthmap_pyr"][lvl] is None:
                continue

            left_occlusion_mask_lvl = losses.get_occlusion_mask(
                inputs["K_pyr"][lvl], inputs["T_right_in_left"],
                outputs["left_idepthmap_pyr"][lvl], outputs["left_idepthmap_mask_pyr"][lvl],
                outputs["right_idepthmap_pyr"][lvl], outputs["right_idepthmap_mask_pyr"][lvl])
            left_occlusion_mask_pyr[lvl] = left_occlusion_mask_lvl

            right_occlusion_mask_lvl = losses.get_occlusion_mask(
                inputs["K_pyr"][lvl], inputs["T_left_in_right"],
                outputs["right_idepthmap_pyr"][lvl], outputs["right_idepthmap_mask_pyr"][lvl],
                outputs["left_idepthmap_pyr"][lvl], outputs["left_idepthmap_mask_pyr"][lvl])
            right_occlusion_mask_pyr[lvl] = right_occlusion_mask_lvl

        predictions["left_occlusion_mask_pyr"] = left_occlusion_mask_pyr
        predictions["right_occlusion_mask_pyr"] = right_occlusion_mask_pyr

        # Groundtruth occlusion masks.
        left_occlusion_mask_true = losses.get_occlusion_mask(
            inputs["K_pyr"][0], inputs["T_right_in_left"],
            inputs["left_idepthmap_true"], outputs["left_idepthmap_mask_pyr"][lvl],
            inputs["right_idepthmap_true"], outputs["right_idepthmap_mask_pyr"][lvl])
        predictions["left_occlusion_mask_true"] = left_occlusion_mask_true

        right_occlusion_mask_true = losses.get_occlusion_mask(
            inputs["K_pyr"][0], inputs["T_left_in_right"],
            inputs["right_idepthmap_true"], outputs["right_idepthmap_mask_pyr"][lvl],
            inputs["left_idepthmap_true"], outputs["left_idepthmap_mask_pyr"][lvl])
        predictions["right_occlusion_mask_true"] = right_occlusion_mask_true

    if params["left_right_factor"] > 0.0:
        # Compute left/right idepthmap consistency losses.
        left_right_loss = losses.left_right_idepthmap_consistency_losses(
            inputs["T_right_in_left"], inputs["T_left_in_right"], inputs["K_pyr"],
            outputs["left_idepthmap_pyr"], predictions["left_occlusion_mask_pyr"],
            outputs["right_idepthmap_pyr"], predictions["right_occlusion_mask_pyr"])
        loss += params["left_right_factor"] * left_right_loss
        loss_dict["left_right_loss"] = left_right_loss

    if params["reconstruction_factor"] > 0.0:
        # Compute reconstruction loss--------------------------------------------
        recon_losses = []

        predictions["left_image_pred_pyr"] = [None for _ in range(len(outputs["left_idepthmap_pyr"]))]
        for lvl in range(len(outputs["left_idepthmap_pyr"])):
            if outputs["left_idepthmap_pyr"][lvl] is None:
                continue

            recon_loss_lvl, left_image_pred_lvl = losses.reconstruction_loss(
                inputs["T_right_in_left"], inputs["K_pyr"][0], inputs["left_image_pyr"][0],
                inputs["right_image_pyr"][0], outputs["left_idepthmap_pyr"][lvl],
                predictions["left_occlusion_mask_pyr"][lvl])

            predictions["left_image_pred_pyr"][lvl] = left_image_pred_lvl
            recon_losses.append(recon_loss_lvl)

        predictions["right_image_pred_pyr"] = [None for _ in range(len(outputs["right_idepthmap_pyr"]))]
        for lvl in range(len(outputs["left_idepthmap_pyr"])):
            if outputs["right_idepthmap_pyr"][lvl] is None:
                continue

            recon_loss_lvl, right_image_pred_lvl = losses.reconstruction_loss(
                inputs["T_left_in_right"], inputs["K_pyr"][0], inputs["right_image_pyr"][0],
                inputs["left_image_pyr"][0], outputs["right_idepthmap_pyr"][lvl],
                predictions["right_occlusion_mask_pyr"][lvl])

            predictions["right_image_pred_pyr"][lvl] = right_image_pred_lvl
            recon_losses.append(recon_loss_lvl)

        loss_dict["reconstruction_losses"] = recon_losses

        recon_loss = sum(recon_losses)
        loss += params["reconstruction_factor"] * recon_loss
        loss_dict["reconstruction_loss"] = recon_loss

    return loss, loss_dict, predictions
