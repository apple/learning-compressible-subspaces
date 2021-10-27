#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import os

import numpy as np
import torch
from torch import nn

import utils
from curve_utils import alpha_bit_map
from models.networks import model_profiling


def save_model_at_epoch(model, epoch, save_dir, save_per=10):
    """
    Saves the given model as 'model_{epoch}'
    """
    if epoch % save_per == 0:
        save_path = os.path.join(save_dir, f"model_{epoch}")
        torch.save(model.state_dict(), save_path)
    else:
        print(
            f"Skipping saving at epoch {epoch} (saving every {save_per} epochs)."
        )


def save(save_name, log, save_dir):
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, log)


def _handle_bn_stats(metric_dict, extra_metrics, regime_params):
    if metric_dict is None:
        return None

    if "bn_metrics" not in extra_metrics:
        print(f"No BN metrics found, skipping BN stats.")
        return None

    bn_metrics = extra_metrics["bn_metrics"]

    for name, value in bn_metrics.items():
        if name not in metric_dict:
            metric_dict[name] = []
        metric_dict[name].append(value)
    return metric_dict


def sparse_logging(
    model,
    loss,
    acc,
    model_type,
    param=None,
    metric_dict=None,
    extra_metrics=None,
    **regime_params,
):
    metric_dict = _handle_bn_stats(metric_dict, extra_metrics, regime_params)

    sparsity = utils.get_sparsity_rate(model)

    param_name = "alpha" if model_type == "curve" else "topk"
    param_print = "" if param is None else f" ({param_name} = {param})"
    print(
        f"Test set{param_print}: Average loss: {loss:.4f} | Accuracy: {acc:.4f} | Sparsity: {sparsity:.4f}"
    )

    if metric_dict is not None:
        metric_dict["acc"].append(acc)
        metric_dict["sparsity"].append(sparsity)
        if param_name not in metric_dict:
            metric_dict[param_name] = []
        metric_dict[param_name].append(param)

    return metric_dict


def quantized_logging(
    model,
    loss,
    acc,
    model_type,
    param=None,
    metric_dict=None,
    extra_metrics=None,
    **regime_params,
):
    metric_dict = _handle_bn_stats(metric_dict, extra_metrics, regime_params)

    if model_type == "curve":
        param_name = "alpha"
        inv_alpha = regime_params.get("inv_alpha", False)
        if inv_alpha:
            num_bits = alpha_bit_map(1 - param, **regime_params)
        else:
            num_bits = alpha_bit_map(param, **regime_params)
        param_print = f" (alpha/num_bits = {param}/{num_bits})"
    else:
        param_name = "num_bits"
        num_bits = param
        param_print = "" if param is None else f" ({param_name} = {num_bits})"

    print(
        f"Test set{param_print}: Average loss: {loss:.4f} | Accuracy: {acc:.4f}"
    )

    if metric_dict is not None:
        metric_dict["acc"].append(acc)
        metric_dict["num_bits"].append(num_bits)
        if param_name != "num_bits":  # Append alpha for curves
            metric_dict[param_name].append(param)

    return metric_dict


def lec_logging(
    model,
    loss,
    acc,
    model_type,
    param=None,
    metric_dict=None,
    extra_metrics=None,
    **regime_params,
):
    metric_dict = _handle_bn_stats(metric_dict, extra_metrics, regime_params)

    sparsity = regime_params["sparsity"]

    param_name = "alpha" if model_type == "curve" else "topk"
    param_print = "" if param is None else f" ({param_name} = {param})"
    print(
        f"Test set{param_print}: Average loss: {loss:.4f} | Accuracy: {acc:.4f} | Sparsity: {sparsity:.4f}"
    )

    if metric_dict is not None:
        metric_dict["acc"].append(acc)
        metric_dict["sparsity"].append(sparsity)
        if param_name not in metric_dict:
            metric_dict[param_name] = []
        metric_dict[param_name].append(param)

    return metric_dict


def ns_logging(
    model,
    loss,
    acc,
    model_type,
    param=None,
    metric_dict=None,
    extra_metrics=None,
    **regime_params,
):
    metric_dict = _handle_bn_stats(metric_dict, extra_metrics, regime_params)

    input_size = regime_params["input_size"]

    is_cuda = next(model.parameters()).is_cuda
    n_macs, n_params = model_profiling.model_profiling(
        model, input_size, input_size, use_cuda=is_cuda
    )

    total_params = 0
    for module in model.modules():
        if isinstance(
            module, (nn.Linear, nn.Conv2d)
        ):  # Make sure we also count contributions from non-sparse elements.
            total_params += module.weight.numel()

    sparsity = (total_params - n_params) / total_params

    param_name = "width_factor"
    param_print = f" ({param_name} = {regime_params[param_name]})"
    print(
        f"Test set{param_print}: Average loss: {loss:.4f} | Accuracy: {acc:.4f} | Sparsity: {sparsity:.4f}"
    )

    if metric_dict is not None:
        metric_dict["acc"].append(acc)
        metric_dict["sparsity"].append(sparsity)
        metric_dict["alpha"].append(param)
        if param_name not in metric_dict:
            metric_dict[param_name] = []
        metric_dict[param_name].append(regime_params[param_name])

    return metric_dict


def us_logging(*args, **kwargs):
    return ns_logging(*args, **kwargs)
