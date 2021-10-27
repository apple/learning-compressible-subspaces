#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import time

import numpy as np
import torch
import torch.nn as nn

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15

num_forwards = 10


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print("Elapsed time: %f ms." % self.time)


def get_params(module):
    """get number of params in module"""
    if hasattr(module, "width_factor"):
        # We assume it's a conv layer.
        ret = module.get_num_parameters()
    else:
        ret = np.sum([np.prod(list(w.size())) for w in module.parameters()])
    return ret


def run_forward(module, input):
    with Timer() as t:
        for _ in range(num_forwards):
            module.forward(*input)
            if input[0].is_cuda:
                torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        "kernel_size": "k",
        "stride": "s",
        "padding": "pad",
        "bias": "b",
        "groups": "g",
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(module, input, output, verbose):
    if not isinstance(input[0], list):
        # Some modules return a list of outputs. We usually ignore them.
        ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(module)
    if isinstance(module, nn.Conv2d):
        module.n_macs = (
            ins[1]
            * outs[1]
            * module.kernel_size[0]
            * module.kernel_size[1]
            * outs[2]
            * outs[3]
            // module.groups
        ) * outs[0]
        module.n_params = get_params(module)
        module.n_seconds = run_forward(module, input)
        module.name = conv_module_name_filter(module.__repr__())
    elif isinstance(module, nn.ConvTranspose2d):
        module.n_macs = (
            ins[1]
            * outs[1]
            * module.kernel_size[0]
            * module.kernel_size[1]
            * outs[2]
            * outs[3]
            // module.groups
        ) * outs[0]
        module.n_params = get_params(module)
        module.n_seconds = run_forward(module, input)
        module.name = conv_module_name_filter(module.__repr__())
    elif isinstance(module, nn.Linear):
        module.n_macs = ins[1] * outs[1] * outs[0]
        module.n_params = get_params(module)
        module.n_seconds = run_forward(module, input)
        module.name = module.__repr__()
    elif isinstance(module, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        module.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        module.n_params = 0
        module.n_seconds = run_forward(module, input)
        module.name = module.__repr__()
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        module.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        module.n_params = 0
        module.n_seconds = run_forward(module, input)
        module.name = module.__repr__()
    else:
        # This works only in depth-first travel of modules.
        module.n_macs = 0
        module.n_params = 0
        module.n_seconds = 0
        num_children = 0
        for m in module.children():
            module.n_macs += getattr(m, "n_macs", 0)
            module.n_params += getattr(m, "n_params", 0)
            module.n_seconds += getattr(m, "n_seconds", 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d,
            nn.InstanceNorm2d,
            nn.Dropout2d,
            nn.Dropout,
            nn.Sequential,
            nn.ReLU6,
            nn.ReLU,
            nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d,
            nn.modules.activation.Sigmoid,
        ]
        if (
            not getattr(module, "ignore_model_profiling", False)
            and module.n_macs == 0
            and t not in ignore_zeros_t
        ):
            print(
                "WARNING: leaf module {} has zero n_macs.".format(type(module))
            )
        return
    if verbose:
        print(
            module.name.ljust(name_space, " ")
            + "{:,}".format(module.n_params).rjust(params_space, " ")
            + "{:,}".format(module.n_macs).rjust(macs_space, " ")
            + "{:,}".format(module.n_seconds).rjust(seconds_space, " ")
        )
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(
            lambda m, input, output: module_profiling(
                m, input, output, verbose=verbose
            )
        )
    )


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(
    model, height, width, batch=1, channel=3, use_cuda=True, verbose=True
):
    """Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    data = torch.rand(batch, channel, height, width)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    print(
        "Item".ljust(name_space, " ")
        + "params".rjust(macs_space, " ")
        + "macs".rjust(macs_space, " ")
        + "nanosecs".rjust(seconds_space, " ")
    )
    if verbose:
        print(
            "".center(
                name_space + params_space + macs_space + seconds_space, "-"
            )
        )
    model(data)
    if verbose:
        print(
            "".center(
                name_space + params_space + macs_space + seconds_space, "-"
            )
        )
    print(
        "Total".ljust(name_space, " ")
        + "{:,}".format(model.n_params).rjust(params_space, " ")
        + "{:,}".format(model.n_macs).rjust(macs_space, " ")
        + "{:,}".format(model.n_seconds).rjust(seconds_space, " ")
    )
    remove_profiling_hooks()
    return model.n_macs, model.n_params
