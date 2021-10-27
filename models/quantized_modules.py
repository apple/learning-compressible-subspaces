#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

# Modified version of https://github.com/pytorch/pytorch/blob/master/torch/nn/intrinsic/qat/modules/conv_fused.py

import math
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from .quantize_affine import QuantizeAffine
from .quantize_affine import quantize_affine

MOD = TypeVar("MOD", bound=nn.modules.conv._ConvNd)


class QuantStandardBN(nn.BatchNorm2d):
    def get_weight(self):
        return self.weight, self.bias


class NoOpBN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):

    _version = 2
    _FLOAT_MODULE = MOD

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        # BatchNormNd args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        dim=2,
        num_bits=8,
        iteration_delay=0,
        bn_module=None,
        norm_kwargs=None,
    ):
        nn.modules.conv._ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            False,
            padding_mode,
        )
        self.freeze_bn = freeze_bn if self.training else True
        if bn_module is not None:
            if "GN" in bn_module.__name__:
                num_groups = norm_kwargs["num_groups"]
                if "Line" in bn_module.__name__:
                    self.bn = bn_module(
                        out_channels, eps=eps, num_groups=num_groups
                    )
                else:
                    self.bn = bn_module(
                        out_channels,
                        eps=eps,
                        num_groups=num_groups,
                        affine=True,
                    )
            else:
                self.bn = bn_module(
                    out_channels, eps=eps, momentum=momentum, affine=True
                )

        else:
            raise ValueError("BN module must be supplied")

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        # Quantization functions/parameters
        self.num_bits = num_bits
        self.iteration_delay = iteration_delay
        self.quantize_input = QuantizeAffine(
            iteration_delay=iteration_delay, num_bits=self.num_bits
        )

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ConvBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def update_num_bits(self, num_bits):
        self.num_bits = num_bits
        self.quantize_input.num_bits = num_bits

    def start_counter(self):
        self.quantize_input.increment_counter = True

    def stop_counter(self):
        self.quantize_input.increment_counter = False

    def is_active(self):
        return self.quantize_input.is_active()

    def _forward(self, input, weight):
        if isinstance(self.bn, nn.BatchNorm2d):
            assert self.bn.running_var is not None
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.get_weight()[0] / running_std
        elif isinstance(self.bn, (nn.InstanceNorm2d, nn.GroupNorm)):
            # We can't really apply the scale factor easily because each batch
            # element is weighted differently. So, we don't fuse the
            # InstanceNorm into the convolution.
            scale_factor = torch.ones(self.out_channels, device=weight.device)

        weight_shape = [1] * len(weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(weight.shape)
        bias_shape[1] = -1
        # Quantize weights
        # Note: weights are quantized from the beginning of training, i.e., no delay here
        scaled_weight = quantize_affine(
            weight * scale_factor.reshape(weight_shape), num_bits=self.num_bits
        )
        # Quantize input
        # Note: inputs are quantized after self.iteration_delay training iterations
        input = self.quantize_input(input)
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(
                self.out_channels, device=scaled_weight.device
            )
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def extra_repr(self):
        return super(_ConvBnNd, self).extra_repr()

    # def forward(self, input):
    #     return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                "bn.weight": "gamma",
                "bn.bias": "beta",
                "bn.running_mean": "running_mean",
                "bn.running_var": "running_var",
                "bn.num_batches_tracked": "num_batches_tracked",
            }

            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_ConvBnNd, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class ConvBn2d(_ConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.
    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.
    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight
    """

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        num_bits=8,
        iteration_delay=0,
        bn_module=None,
        norm_kwargs=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            dim=2,
            num_bits=num_bits,
            iteration_delay=iteration_delay,
            bn_module=bn_module,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, input, weight=None):
        if weight is None:
            return _ConvBnNd._forward(self, input, self.weight)
        else:
            return _ConvBnNd._forward(self, input, weight)


class QuantSubspaceConvBn2d(ConvBn2d):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the
        # corresponding weight.
        w = self.get_weight()
        return super().forward(x, w)


class TwoParamConvBnd2d(QuantSubspaceConvBn2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight)
        initialize_fn(self.weight1)


class LinesConvBn2d(TwoParamConvBnd2d):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w
