# Copyright 2021 Massachusetts Institute of Technology
#
# @file resnet.py
# @author W. Nicholas Greene
# @date 2020-04-24 12:14:10 (Fri)

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):
    """Residual block implementation from torchvision, customized to support dilated
    convolutions.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groups, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SimpleBasicBlock(nn.Module):
    """Residual block implementation from torchvision, customized to support dilated
    convolutions.

    Based off the residual block from here:
    https://github.com/meteorshowers/StereoNet-ActiveStereoNet

    Uses biases in conv layers.
    Uses just 1 convolution instead of 2.
    Does not apply final activation.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bias=True):
        super(SimpleBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groups, dilation, bias)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, 1, groups, dilation)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        # out = self.relu(out) # BasicBlock in meteorshowers StereoNet doesn't have last activation.

        return out

class SimpleBasicBlock3D(nn.Module):
    """Residual block implementation from torchvision, customized to support dilated
    convolutions (3D version).

    Based off the residual block from here:
    https://github.com/meteorshowers/StereoNet-ActiveStereoNet

    Uses biases in conv layers.
    Uses just 1 convolution instead of 2.
    Does not apply final activation.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SimpleBasicBlock3D, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=True, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, 1, groups, dilation)
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        # out = self.relu(out) # BasicBlock in meteorshowers StereoNet doesn't have last activation.

        return out
