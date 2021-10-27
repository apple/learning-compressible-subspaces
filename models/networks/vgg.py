#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import math

import torch.nn as nn

__all__ = ["vgg", "vgg11", "vgg13", "vgg16", "vgg19"]

defaultcfg = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
}


class vgg(nn.Module):
    def __init__(
        self,
        dataset="imagenet",
        depth=19,
        init_weights=True,
        cfg=None,
        builder=None,
        block_builder=None,
    ):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(
            cfg, True, builder=builder, block_builder=block_builder
        )

        if dataset == "imagenet":
            num_classes = 1000
        else:
            raise NotImplementedError(f"Not implemented for dataset {dataset}")
        self.classifier = builder.conv1x1(cfg[-1], num_classes, last_layer=True)

        if init_weights:
            self._initialize_weights()

    def make_layers(
        self, cfg, batch_norm=False, builder=None, block_builder=None
    ):
        layers = []
        in_channels = 3
        first_conv_layer = True
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if first_conv_layer:
                    my_builder = builder
                else:
                    my_builder = block_builder
                conv2d = my_builder.conv3x3(
                    in_channels, v, first_layer=first_conv_layer
                )
                first_conv_layer = False
                if batch_norm:
                    bn = my_builder.batchnorm(v)
                    layers += [conv2d, bn, nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        y = self.classifier(x)
        y = y.view(y.size(0), -1)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.last_layer:
                    assert m.kernel_size == (1, 1)
                    # Fully Connected Layer
                    i = 1
                    while hasattr(m, f"weight{i}"):
                        weight = getattr(m, f"weight{i}")
                        weight.data.normal_(0, 0.01)
                        bias = getattr(m, f"bias{i}", None)
                        if bias is not None:
                            bias.data.zero_()
                        i += 1
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    i = 1
                    while hasattr(m, f"weight{i}"):
                        weight = getattr(m, f"weight{i}")
                        weight.data.normal_(0, math.sqrt(2.0 / n))
                        bias = getattr(m, f"bias{i}", None)
                        if bias is not None:
                            bias.data.zero_()
                        i += 1

            elif isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
                i = 1
                while hasattr(m, f"weight{i}"):
                    weight = getattr(m, f"weight{i}")
                    weight.data.fill_(0.5)
                    bias = getattr(m, f"bias{i}", None)
                    if bias is not None:
                        bias.data.zero_()
                    i += 1
            elif isinstance(m, nn.Linear):
                i = 1
                while hasattr(m, f"weight{i}"):
                    weight = getattr(m, f"weight{i}")
                    weight.data.normal_(0, 0.01)
                    bias = getattr(m, f"bias{i}", None)
                    if bias is not None:
                        bias.data.zero_()
                    i += 1


def vgg11(**kwargs):
    return vgg(depth=11, **kwargs)


def vgg13(**kwargs):
    return vgg(depth=13, **kwargs)


def vgg16(**kwargs):
    return vgg(depth=16, **kwargs)


def vgg19(**kwargs):
    return vgg(depth=19, **kwargs)
