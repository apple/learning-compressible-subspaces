#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutions
StandardConv = nn.Conv2d


class SubspaceConv(nn.Conv2d):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the
        # corresponding weight.
        w = self.get_weight()
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight)
        initialize_fn(self.weight1)


class LinesConv(TwoParamConv):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w


# BatchNorms
StandardBN = nn.BatchNorm2d


class SubspaceBN(nn.BatchNorm2d):
    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the
        # corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None
            )
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be
            # updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class TwoParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.empty([self.num_features]))
        self.bias1 = nn.Parameter(torch.empty([self.num_features]))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)


class LinesBN(TwoParamBN):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


# InstanceNorm
def StandardIN(*args, affine=True, **kwargs):
    return nn.InstanceNorm2d(*args, affine=affine, **kwargs)


def _process_num_groups(num_groups: Union[str, int], num_channels: int) -> int:
    if num_groups == "full":
        num_groups = num_channels  # Set it equal to num_features.
    else:
        num_groups = int(num_groups)

    # If num_groups is greater than num_features, we reduce it.
    num_groups = min(num_channels, num_groups)
    return num_groups


class SubspaceIN(nn.InstanceNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        # Override @affine to be true.
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=track_running_stats,
        )

    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the
        # corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for instancenorm.
        assert self.running_mean is None or isinstance(
            self.running_mean, torch.Tensor
        )
        assert self.running_var is None or isinstance(
            self.running_var, torch.Tensor
        )
        return F.instance_norm(
            input,
            self.running_mean,
            self.running_var,
            w,
            b,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class TwoParamIN(SubspaceIN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.empty([self.num_features]))
        self.bias1 = nn.Parameter(torch.empty([self.num_features]))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)


class LinesIN(TwoParamIN):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


# GroupNorm
def StandardGN(*args, affine=True, **kwargs):
    num_groups = kwargs.pop("num_groups", "full")
    num_groups = _process_num_groups(num_groups, args[0])
    return nn.GroupNorm(num_groups, *args, affine=affine, **kwargs)


class SubspaceGN(nn.GroupNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        *,
        num_groups: Union[str, int],
    ) -> None:

        num_groups = _process_num_groups(num_groups, num_features)

        # Override @affine to be true.
        super().__init__(
            num_groups,
            num_features,
            eps=eps,
            affine=True,
        )
        self.num_features = num_features

    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the
        # corresponding weight.
        w, b = self.get_weight()
        return F.group_norm(input, self.num_groups, w, b, self.eps)


class TwoParamGN(SubspaceGN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.empty([self.num_features]))
        self.bias1 = nn.Parameter(torch.empty([self.num_features]))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)


class LinesGN(TwoParamGN):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


def _get_num_parameters(conv):
    in_channels = conv.in_channels
    out_channels = conv.out_channels

    if hasattr(conv, "in_channels_list"):
        in_channels_ratio = in_channels / max(conv.in_channels_list)
        out_channels_ratio = out_channels / max(conv.out_channels_list)
    else:
        in_channels_ratio = in_channels / conv.in_channels_max
        out_channels_ratio = out_channels / conv.out_channels_max

    ret = conv.weight.numel()
    ret = max(1, round(ret * in_channels_ratio * out_channels_ratio))
    return ret


# Adaptive modules (which adjust their number of channels at inference time).

# This code contains the norm implementation used in our unstructured sparsity
# experiments and baselines. Note that normally, we disallow storing or
# recomputing BatchNorm statistics. However, we retain the ability to store
# individual BatchNorm statistics purely for sanity-checking purposes (to ensure
# our implementation produces similar results to the Universal Slimming paper,
# when BatchNorms are stored). But, we don't use these results in any analysis.
class AdaptiveNorm(nn.modules.batchnorm._NormBase):
    def __init__(
        self,
        bn_class,
        bn_func,
        mode,
        *args,
        ratio=1,
        width_factors_list=None,
        **kwargs,
    ):
        assert mode in ("BatchNorm", "InstanceNorm", "GroupNorm")

        kwargs_cpy = kwargs.copy()
        try:
            track_running_stats = kwargs_cpy.pop("track_running_stats")
        except KeyError:
            track_running_stats = False

        try:
            self.num_groups = kwargs_cpy.pop("num_groups")
        except KeyError:
            self.num_groups = None

        super().__init__(
            *args,
            affine=True,
            track_running_stats=track_running_stats,
            **kwargs_cpy,
        )

        num_features = args[0]
        self.width_factors_list = width_factors_list
        self.num_features_max = num_features
        if mode == "BatchNorm" and self.width_factors_list is not None:
            print(
                f"Storing extra BatchNorm layers. This should only be used"
                f"for sanity checking, since it violates our goal of"
                f"arbitrarily fine-grained compression levels at inference"
                f"time."
            )
            self.bn = nn.ModuleList(
                [
                    bn_class(i, affine=False)
                    for i in [
                        max(1, round(self.num_features_max * width_factor))
                        for width_factor in self.width_factors_list
                    ]
                ]
            )
        if mode == "GroupNorm":
            if self.num_groups is None:
                raise ValueError("num_groups is required")
            if self.num_groups not in ("full", 1):
                # This must be "full" or 1, or the tensor might not be divisible
                # by @self.num_groups.
                raise ValueError(f"Invalid num_groups={self.num_groups}")

        self.ratio = ratio
        self.width_factor = None
        self.ignore_model_profiling = True
        self.bn_func = bn_func
        self.mode = mode

    def get_weight(self):
        return self.weight, self.bias

    def forward(self, input):
        weight, bias = self.get_weight()
        c = input.shape[1]
        if (
            self.mode == "BatchNorm"
            and self.width_factors_list is not None
            and self.width_factor in self.width_factors_list
        ):
            # Normally, we expect width_factors_list to be empty, because we
            # only want to use it if we are running sanity checks (e.g.
            # recreating the original performance or something).
            idx = self.width_factors_list.index(self.width_factor)
            kwargs = {
                "input": input,
                "running_mean": self.bn[idx].running_mean[:c],
                "running_var": self.bn[idx].running_var[:c],
                "weight": weight[:c],
                "bias": bias[:c],
                "training": self.training,
                "momentum": self.momentum,
                "eps": self.eps,
            }
        elif self.mode in ("InstanceNorm", "BatchNorm"):
            # Sanity check, since we're not tracking running stats.
            running_mean = self.running_mean
            if self.running_mean is not None:
                running_mean = running_mean[:c]

            running_var = self.running_var
            if self.running_var is not None:
                running_var = running_var[:c]

            kwargs = {
                "input": input,
                "running_mean": running_mean,
                "running_var": running_var,
                "weight": weight[:c],
                "bias": bias[:c],
                "momentum": self.momentum,
                "eps": self.eps,
            }

            if self.mode == "BatchNorm":
                kwargs["training"] = self.training

        elif self.mode == "GroupNorm":
            num_groups = self.num_groups
            if num_groups == "full":
                num_groups = c
            kwargs = {
                "input": input,
                "num_groups": num_groups,
                "weight": weight[:c],
                "bias": bias[:c],
                "eps": self.eps,
            }
        else:
            raise NotImplementedError(f"Invalid mode {self.mode}.")

        return self.bn_func(**kwargs)


class AdaptiveBN(AdaptiveNorm):
    def __init__(self, *args, **kwargs):
        norm_class = nn.BatchNorm2d
        norm_func = F.batch_norm
        super().__init__(norm_class, norm_func, "BatchNorm", *args, **kwargs)


class AdaptiveIN(AdaptiveNorm):
    def __init__(self, *args, **kwargs):
        norm_class = nn.InstanceNorm2d
        norm_func = F.instance_norm
        super().__init__(norm_class, norm_func, "InstanceNorm", *args, **kwargs)


class LinesAdaptiveIN(AdaptiveIN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)

    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b


class AdaptiveConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        first_layer=False,
        last_layer=False,
        ratio=None,
    ):
        self.first_layer = first_layer
        self.last_layer = last_layer

        if ratio is None:
            ratio = [1, 1]

        super(AdaptiveConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if groups == in_channels:
            assert in_channels == out_channels
            self.depthwise = True
        else:
            self.depthwise = False

        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_factor = None
        self.ratio = ratio

    def get_weight(self):
        return self.weight

    def forward(self, input):
        if not self.first_layer:
            self.in_channels = input.shape[1]
        if not self.last_layer:
            self.out_channels = max(
                1, round(self.out_channels_max * self.width_factor)
            )
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.get_weight()
        weight = weight[: self.out_channels, : self.in_channels, :, :]
        assert self.bias is None
        bias = None
        y = nn.functional.conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return y

    def get_num_parameters(self):
        return _get_num_parameters(self)


class LinesAdaptiveConv2d(AdaptiveConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.empty_like(self.weight))
        assert self.bias is None
        torch.nn.init.ones_(self.weight1)

    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w
