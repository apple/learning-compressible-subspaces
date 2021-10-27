#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
from __future__ import absolute_import

import math

import torch.nn as nn

from .channel_selection import channel_selection

__all__ = ["cpreresnet"]

"""
Preactivation resnet with bottleneck design.
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        cfg,
        stride=1,
        downsample=None,
        builder=None,
        channel_selection_active=True,
    ):
        super(Bottleneck, self).__init__()
        if builder is None:
            raise ValueError(f"Builder required, got None")

        self.bn1 = builder.batchnorm(inplanes)
        self.select = channel_selection(
            inplanes, active=channel_selection_active
        )
        self.conv1 = builder.conv1x1(cfg[0], cfg[1])
        self.bn2 = builder.batchnorm(cfg[1])
        self.conv2 = builder.conv3x3(cfg[1], cfg[2], stride=stride)
        self.bn3 = builder.batchnorm(cfg[2])
        self.conv3 = builder.conv1x1(cfg[2], planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class cpreresnet(nn.Module):
    def __init__(
        self,
        depth=20,
        dataset="cifar10",
        cfg=None,
        builder=None,
        block_builder=None,
        channel_selection_active=True,
    ):
        super(cpreresnet, self).__init__()
        # assert block_builder is None, "Should not provide a block_builder."
        if builder is None:
            raise ValueError(f"Expected builder, got None.")
        assert (depth - 2) % 9 == 0, "depth should be 9n+2"

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [
                [16, 16, 16],
                [64, 16, 16] * (n - 1),
                [64, 32, 32],
                [128, 32, 32] * (n - 1),
                [128, 64, 64],
                [256, 64, 64] * (n - 1),
                [256],
            ]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = builder.conv3x3(3, 16, first_layer=True)
        self.layer1 = self._make_layer(
            block,
            16,
            n,
            cfg=cfg[0 : 3 * n],
            builder=block_builder,
            channel_selection_active=channel_selection_active,
        )
        self.layer2 = self._make_layer(
            block,
            32,
            n,
            cfg=cfg[3 * n : 6 * n],
            stride=2,
            builder=block_builder,
            channel_selection_active=channel_selection_active,
        )
        self.layer3 = self._make_layer(
            block,
            64,
            n,
            cfg=cfg[6 * n : 9 * n],
            stride=2,
            builder=block_builder,
            channel_selection_active=channel_selection_active,
        )
        self.bn = builder.batchnorm(64 * block.expansion)
        self.select = channel_selection(
            64 * block.expansion, active=channel_selection_active
        )
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        # We work only with CIFAR-10
        num_categories = 10

        self.fc = builder.conv1x1(cfg[-1], num_categories, last_layer=True)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m != self.fc:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                i = 1
                while hasattr(m, f"weight{i}"):
                    weight = getattr(m, f"weight{i}")
                    weight.data.normal_(0, math.sqrt(2.0 / n))
                    i += 1
            elif isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
                i = 1
                while hasattr(m, f"weight{i}"):
                    weight = getattr(m, f"weight{i}")
                    weight.data.fill_(0.5)
                    bias = getattr(m, f"bias{i}")
                    bias.data.zero_()
                    i += 1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        cfg,
        stride=1,
        builder=None,
        channel_selection_active=True,
    ):
        if builder is None:
            raise ValueError(f"Expected builder, got None.")
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                builder.conv1x1(
                    self.inplanes, planes * block.expansion, stride=stride
                ),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                cfg[0:3],
                stride,
                downsample,
                builder=builder,
                channel_selection_active=channel_selection_active,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    cfg[3 * i : 3 * (i + 1)],
                    builder=builder,
                    channel_selection_active=channel_selection_active,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        assert x.shape[2:] == (1, 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x
