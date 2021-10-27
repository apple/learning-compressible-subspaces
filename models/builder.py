#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

from . import init
from . import modules
from . import quantized_modules
from . import sparse_modules


class Builder:
    def __init__(
        self,
        conv_type="LinesConv",
        bn_type="LinesBN",
        conv_init="kaiming_normal",
        norm_kwargs=None,
        pass_first_last=False,
        **conv_kwargs
    ):
        if norm_kwargs is None:
            norm_kwargs = {}
        self.pass_first_last = pass_first_last

        self.conv_kwargs = conv_kwargs
        self.norm_kwargs = norm_kwargs

        if hasattr(modules, bn_type):
            self.bn_layer = getattr(modules, bn_type)
        elif hasattr(sparse_modules, bn_type):
            self.bn_layer = getattr(sparse_modules, bn_type)
        elif hasattr(quantized_modules, bn_type):
            self.bn_layer = getattr(quantized_modules, bn_type)
        else:
            raise ValueError("Normalization layer not found")

        if hasattr(modules, conv_type):
            self.conv_layer = getattr(modules, conv_type)
        elif hasattr(sparse_modules, conv_type):
            self.conv_layer = getattr(sparse_modules, conv_type)
        elif hasattr(quantized_modules, conv_type):
            self.conv_layer = getattr(quantized_modules, conv_type)
            self.conv_kwargs[
                "bn_module"
            ] = self.bn_layer  # self.bn_layer chosen above
            self.conv_kwargs["norm_kwargs"] = norm_kwargs
            # Overwrite self.bn_layer to no-op batchnorm since handled by ConvBn
            self.bn_layer = getattr(quantized_modules, "NoOpBN")
        else:
            raise ValueError("Convolution layer not found")

        self.conv_init = getattr(init, conv_init)

    def conv(
        self,
        kernel_size,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        conv_kwargs = self.conv_kwargs.copy()
        if self.pass_first_last:
            conv_kwargs["first_layer"] = first_layer
            conv_kwargs["last_layer"] = last_layer

        if kernel_size == 1:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                **conv_kwargs
            )
        elif kernel_size == 3:
            conv = self.conv_layer(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
                **conv_kwargs
            )
        elif kernel_size == 5:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                groups=groups,
                bias=False,
                **conv_kwargs
            )
        elif kernel_size == 7:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                groups=groups,
                bias=False,
                **conv_kwargs
            )
        else:
            return None

        conv.first_layer = first_layer
        conv.last_layer = last_layer
        conv.is_conv = is_conv
        self.conv_init(conv.weight)
        if hasattr(conv, "initialize"):
            conv.initialize(self.conv_init)
        return conv

    def conv1x1(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """1x1 convolution with padding"""
        c = self.conv(
            1,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )

        return c

    def conv3x3(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """3x3 convolution with padding"""
        c = self.conv(
            3,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )
        return c

    def conv5x5(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """5x5 convolution with padding"""
        c = self.conv(
            5,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )
        return c

    def conv7x7(
        self,
        in_planes,
        out_planes,
        stride=1,
        groups=1,
        first_layer=False,
        last_layer=False,
        is_conv=False,
    ):
        """7x7 convolution with padding"""
        c = self.conv(
            7,
            in_planes,
            out_planes,
            stride=stride,
            groups=groups,
            first_layer=first_layer,
            last_layer=last_layer,
            is_conv=is_conv,
        )
        return c

    def batchnorm(self, planes):
        return self.bn_layer(planes, **self.norm_kwargs)
