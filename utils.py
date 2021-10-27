#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import collections
import os
import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml

import schedulers
from models import modules
from models import networks
from models.builder import Builder
from models.sparse_modules import SparseConv2d


def get_cifar10_data(batch_size):
    print("==> Preparing CIFAR-10 data...")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    # Note that, we perform an analysis of BatchNorm statistics when validating,
    # so we *must* shuffle the validation set.
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    return trainset, trainloader, testset, testloader


def get_imagenet_data(data_dir, batch_size):
    print("==> Preparing ImageNet data...")

    traindir = os.path.join(data_dir, "training")
    valdir = os.path.join(data_dir, "validation")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    trainset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    testset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    # Note that, we perform an analysis of BatchNorm statistics when validating,
    # so we *must* shuffle the validation set.
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    return trainset, trainloader, testset, testloader


def get_yaml_config(config_file):
    print(f"Reading config {config_file}")
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def print_train_params(config, setting, method, norm, save_dir):
    params = config["parameters"]

    model = params["model_config"]["model_class"]
    dataset = params["dataset"]

    model_name_map = {
        "cpreresnet20": "cPreResNet20",
        "resnet18": "ResNet18",
        "vgg19": "VGG19",
    }

    method_name_map = {
        "lcs_l": "LCS+L",
        "lcs_p": "LCS+P",
        "ns": "NS",
        "us": "US",
        "lec": "LEC",
        "target_topk": "TopK Target",
        "target_bit_width": "Bit Width Target",
    }

    dataset_name_map = {"cifar10": "CIFAR-10", "imagenet": "ImageNet"}

    setting_name_map = {
        "unstructured_sparsity": "Unstructured sparsity",
        "structured_sparsity": "Structured sparsity",
        "quantized": "Quantized",
    }

    msg = f"{setting_name_map[setting]} ({method_name_map[method]}) training:"
    msg += f" {model_name_map[model]} on {dataset_name_map[dataset]} w/ {norm}."

    if method == "target_topk":
        topk_target = params["topk"]
        msg += f" TopK target: {topk_target}."
    elif method == "target_bit_width":
        bit_width_target = params["num_bits"]
        msg += f" Bit width target: {bit_width_target}."

    print()
    print(msg)
    print(f"Saving to {save_dir}.")
    print()
    time.sleep(5)


def create_save_dir(base_save, method, model, dataset, norm, use_default=True):
    if use_default:
        save_dir = f"{base_save}/{method}/{model}/{dataset}/{norm}"
    else:
        save_dir = base_save

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    return save_dir


def network_and_params(config=None):
    """
    Returns the network and training parameters for the specified model type.
    """

    model_config = config["parameters"].get("model_config")
    model_class = model_config["model_class"]
    model_class_dict = {
        "cpreresnet20": networks.cpreresnet,
        "resnet18": networks.ResNet18,
        "vgg19": networks.vgg19,
    }
    if model_config is not None:
        # Get the class.
        if model_class in model_class_dict:
            model_class = model_class_dict[model_class]
        else:
            raise NotImplementedError(
                f"Invalid model_class={model_config['model_class']}"
            )
        if "model_kwargs" in model_config:
            extra_model_kwargs = model_config["model_kwargs"]
        else:
            extra_model_kwargs = {}
    else:
        extra_model_kwargs = {}

    # General params
    epochs = config["parameters"].get("epochs", 200)

    test_freq = config["parameters"].get("test_freq", 20)
    batch_size = config["parameters"].get("batch_size", 128)
    learning_rate = config["parameters"].get("learning_rate", 0.01)
    momentum = config["parameters"].get("momentum", 0.9)
    weight_decay = config["parameters"].get("weight_decay", 0.0005)
    warmup_budget = config["parameters"].get("warmup_budget", 80) / 100.0
    dataset = config["parameters"].get("dataset", "cifar10")
    alpha_grid = config["parameters"].get("alpha_grid", None)

    regime = config["parameters"]["regime"]

    if dataset == "cifar10":
        data = get_cifar10_data(batch_size)
    elif dataset == "imagenet":
        imagenet_dir = config["parameters"]["dataset_dir"]
        data = get_imagenet_data(imagenet_dir, batch_size)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    train_size = len(data[0])
    warmup_iterations = np.ceil(
        warmup_budget * epochs * train_size / batch_size
    )

    # Get model layers
    conv_type = config["parameters"]["conv_type"]
    bn_type = config["parameters"]["bn_type"]
    block_conv_type = config["parameters"]["block_conv_type"]
    block_bn_type = config["parameters"]["block_bn_type"]

    # Get regime-specific parameters
    regime_params = config["parameters"].get("regime_params", {})
    regime_params["regime"] = regime
    builder_parms = {}
    block_builder_params = {}

    # Append dataset to extra_model_kwargs args
    extra_model_kwargs["dataset"] = dataset

    if regime == "sparse":
        if "Sparse" not in block_conv_type:
            raise ValueError(
                "Regime set to sparse but non-sparse convolution layer received..."
            )
        regime_params["topk"] = config["parameters"].get("topk", 0.0)
        regime_params["current_iteration"] = 0
        regime_params["warmup_iterations"] = warmup_iterations
        regime_params["alpha_sampling"] = config["parameters"].get(
            "alpha_sampling", [0.025, 1, 0]
        )

        method = config["parameters"].get("method", "topk")
        block_builder_params["method"] = method
        if "Sparse" in conv_type:
            builder_parms["method"] = method

    elif regime == "lec":
        regime_params["topk"] = config["parameters"].get("topk", 0.0)
        regime_params["current_iteration"] = 0
        regime_params["warmup_iterations"] = warmup_iterations
        regime_params["alpha_sampling"] = config["parameters"].get(
            "alpha_sampling", [0, 1, 0]
        )
        regime_params["model_kwargs"] = {
            "dataset": dataset,
            **extra_model_kwargs,
        }
        regime_params["bn_update_factor"] = config["parameters"].get(
            "bn_update_factor", 0
        )

        regime_params["bn_type"] = config["parameters"]["bn_type"]

    elif regime == "ns":
        width_factors_list = config["parameters"]["builder_kwargs"][
            "width_factors_list"
        ]
        regime_params["width_factors_list"] = width_factors_list

        builder_parms["pass_first_last"] = True

        block_builder_params["pass_first_last"] = True

        if config["parameters"]["block_conv_type"] != "AdaptiveConv2d":
            block_builder_params["width_factors_list"] = width_factors_list
            builder_parms["width_factors_list"] = width_factors_list

        if "BN" in config["parameters"]["bn_type"]:
            norm_kwargs = config["parameters"].get("norm_kwargs", {})

            builder_parms["norm_kwargs"] = {
                "width_factors_list": width_factors_list,
                **norm_kwargs,
            }
            block_builder_params["norm_kwargs"] = {
                "width_factors_list": width_factors_list,
                **norm_kwargs,
            }

        regime_params["bn_type"] = config["parameters"]["bn_type"]

    elif regime == "us":
        builder_parms["pass_first_last"] = True
        block_builder_params["pass_first_last"] = True

        if "BN" in config["parameters"]["bn_type"]:
            norm_kwargs = config["parameters"]["norm_kwargs"]
            assert "width_factors_list" in norm_kwargs

            block_builder_params["norm_kwargs"] = norm_kwargs
            builder_parms["norm_kwargs"] = norm_kwargs

    elif regime == "quantized":
        if "ConvBn2d" not in block_conv_type:
            raise ValueError(
                "Regime set to quanitzed but non-quantized convolution layer received..."
            )
        block_builder_params["num_bits"] = config["parameters"].get(
            "num_bits", 8
        )
        block_builder_params["iteration_delay"] = warmup_iterations

        if conv_type == "ConvBn2d":
            builder_parms["num_bits"] = config["parameters"].get("num_bits", 8)
            builder_parms["iteration_delay"] = warmup_iterations

        regime_params["min_bits"] = config["parameters"].get("min_bits", 2)
        regime_params["max_bits"] = config["parameters"].get("max_bits", 8)
        regime_params["num_bits"] = config["parameters"].get("num_bits", 8)
        regime_params["discrete"] = config["parameters"].get(
            "discrete_alpha_map", False
        )

    regime_params["is_standard"] = config["parameters"].get(
        "is_standard", False
    )
    regime_params["random_param"] = config["parameters"].get(
        "random_param", False
    )
    regime_params["num_points"] = config["parameters"].get("num_points", 0)

    # Evaluation parameters for independent models
    regime_params["eval_param_grid"] = config["parameters"].get(
        "eval_param_grid", None
    )

    # If norm_kwargs haven't been set, and they are present, add them to
    # builder_params and block_builder_params.
    if "norm_kwargs" not in builder_parms:
        builder_parms["norm_kwargs"] = config["parameters"].get(
            "norm_kwargs", {}
        )
    if "norm_kwargs" not in block_builder_params:
        block_builder_params["norm_kwargs"] = config["parameters"].get(
            "norm_kwargs", {}
        )

    # Construct network
    builder = Builder(conv_type=conv_type, bn_type=bn_type, **builder_parms)
    block_builder = Builder(
        block_conv_type, block_bn_type, **block_builder_params
    )

    net = model_class(
        builder=builder, block_builder=block_builder, **extra_model_kwargs
    )

    # Input size
    regime_params["input_size"] = get_input_size(dataset)

    # Save directory
    regime_params["save_dir"] = config["parameters"]["save_dir"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = schedulers.cosine_lr(
        optimizer, learning_rate, warmup_length=5, epochs=epochs
    )

    train_params = [epochs, test_freq, alpha_grid]
    opt_params = [criterion, optimizer, scheduler]

    print(f"Got network:\n{net}")

    return net, opt_params, train_params, regime_params, data


def get_net(model_dir, net, num_gpus=1):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        cudnn.benchmark = True
        state_dict = torch.load(model_dir)
    else:
        state_dict = torch.load(model_dir, map_location=torch.device("cpu"))

    # Set device ids for DataParallel.
    if device == "cuda" and num_gpus > 0:
        device_ids = list(range(num_gpus))
    else:
        device_ids = None
    network = torch.nn.DataParallel(net, device_ids=device_ids)
    network.load_state_dict(state_dict)

    return network


def get_sparsity_rate(model) -> float:
    total_params = 0
    sparse_params = 0
    for module in model.modules():
        if isinstance(module, SparseConv2d):
            sparse_weight = module.apply_sparsity(module.weight)
            total_params += sparse_weight.numel()
            sparse_params += (
                (sparse_weight == 0).float().sum().item()
            )  # changed fuse weight to sparse weight
        elif isinstance(
            module, (nn.Linear, nn.Conv2d)
        ):  # Make sure we also count contributions from non-sparse elements.
            total_params += module.weight.numel()
        # Note: bias parameters are a drop in the bucket.
    sparsity_rate = sparse_params / np.max([total_params, 1])
    return sparsity_rate


def apply_sparsity(
    topk, weight: torch.Tensor, return_scale_factors=False, *, method
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if method == "topk":
        return apply_topk(
            topk, weight, return_scale_factors=return_scale_factors
        )
    else:
        raise NotImplementedError(f"Invalid sparsity method {method}")


def apply_topk(topk: float, weight: torch.Tensor, return_scale_factors=False):
    """
    Given a weight tensor, retains the top self.current_topk of the weights and
    multiplies the rest by 0
    Inputs:
    - weight: A weight tensor, e.g., self.weight
    """
    # Retain only the topk weights, multiplying the rest by 0.
    frac_to_zero = 1 - topk
    with torch.no_grad():
        flat_weight = weight.flatten()
        # Want to convert it away from a special tensor, hence the float() call.
        _, idx = flat_weight.float().abs().sort()
        # @idx is a @special_tensors._SpecialTensor, but we need to convert it
        # to a normal tensor for indexing to work properly.
        idx = torch.tensor(idx, requires_grad=False)
        f = int(frac_to_zero * weight.numel())
        scale_factors = torch.ones_like(flat_weight, requires_grad=False)
        scale_factors[idx[:f]] = 0
        scale_factors = scale_factors.view_as(weight)

    ret = weight * scale_factors

    if return_scale_factors:
        return ret, scale_factors

    return ret


def apply_in_topk_reg(model, apply_to_bias=False):
    loss = None
    count = 0
    for m in model.modules():
        if isinstance(m, modules.SubspaceIN):
            count += 1
            w, b = m.get_weight()

            if loss is None:
                loss = torch.tensor(0.0, device=w.device)
            loss += w.abs().sum()

            if apply_to_bias:
                loss += b.abs().sum()

    if count == 0:
        raise ValueError(f"Didn't adjust any Norms")

    return loss


def get_norm_type_string(model):
    bn_count = 0
    in_count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_count += 1
        elif isinstance(m, nn.InstanceNorm2d):
            in_count += 1

    if bn_count > 0:
        assert in_count == 0, "Got both BN and IN"
        return "StandardBN"
    elif in_count > 0:
        assert bn_count == 0, "Got both BN and IN"
        return "StandardIN"
    else:
        raise ValueError(f"No norm layers detected.")


def make_fresh_copy_of_pruned_network(model: nn.Module, model_kwargs: Dict):
    norm_type_string = get_norm_type_string(model)
    builder = Builder(conv_type="StandardConv", bn_type=norm_type_string)

    copy = type(model.module)(
        builder=builder, block_builder=builder, **model_kwargs
    )  # type: nn.Module
    # Need to move @copy to GPU before moving to DataParallel.
    if next(model.parameters()).is_cuda:
        copy = copy.cuda()
    copy = nn.DataParallel(copy)

    state_dict = model.state_dict()
    del_me = []
    for k, v in state_dict.items():
        if k.endswith(f"1"):
            del_me.append(k)

    for elem in del_me:
        del state_dict[elem]

    copy.load_state_dict(state_dict)

    # The only part we should need to fix are modules with a get_weight()
    # function.
    name_to_copy = {name: module for name, module in copy.named_modules()}

    for name, module in model.named_modules():
        if hasattr(module, "get_weight"):
            print(f"Adjusting weight at module {name}")

            pieces = module.get_weight()

            if len(pieces) == 1:
                name_to_copy[name].weight.data = pieces
            else:
                assert len(pieces) == 2, f"Invalid len(pieces)={len(pieces)}"
                name_to_copy[name].weight.data = pieces[0]
                name_to_copy[name].bias.data = pieces[1]

    return copy


def get_input_size(dataset):
    return {
        "cifar10": 32,
        "imagenet": 224,
    }[dataset]


def register_bn_tracking_hook(module, mean_dict, var_dict, name):
    def forward_hook(_, x, __):
        x = x[0]
        reshaped_x = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)

        mean_dict[name].append(reshaped_x.mean(dim=1).detach().cpu())
        var_dict[name].append(reshaped_x.var(dim=1).detach().cpu())

    return module.register_forward_hook(forward_hook)


def register_bn_tracking_hooks(model: nn.Module):
    mean_dict = collections.defaultdict(list)
    var_dict = collections.defaultdict(list)

    hooks = []
    for name, module in model.named_modules():
        # NOTE: we omit GroupNorm from this check, because it doesn't track
        # running stats (there's no option to do so in PyTorch).
        if (
            isinstance(module, nn.modules.batchnorm._NormBase)
            and module.track_running_stats
        ):
            hooks.append(
                register_bn_tracking_hook(module, mean_dict, var_dict, name)
            )
    return hooks, mean_dict, var_dict


def unregister_bn_tracking_hooks(hooks: List[Callable]):
    for hook in hooks:
        hook.remove()


def get_bn_accuracy_metrics(model: nn.Module, mean_dict: Dict, var_dict: Dict):
    """Determine how accurate the running_mean and running_var of the BatchNorms
    are.

    Note that, the test set should be shuffled during training, or these
    statistics won't be valid.

    Arguments:
        model: the network.
        mean_dict: A dictionary that looks like:
            {'layer_name': [batch1_mean, batch2_mean, ...]}
        var_dict: Similar to mean_dict, but with variances.
    """
    mean_results = collections.defaultdict(list)
    var_results = collections.defaultdict(list)

    name_to_module = {name: module for name, module in model.named_modules()}

    for name in mean_dict.keys():
        module = name_to_module[name]

        running_mean = module.running_mean.detach().cpu()
        assert isinstance(mean_dict[name], list)
        for batch_result in mean_dict[name]:
            num_channels = batch_result.shape[0]
            mean_abs_diff = (
                (batch_result - running_mean[:num_channels]).abs().mean().item()
            )
            mean_results[name].append(mean_abs_diff)

        running_var = module.running_var.detach().cpu()
        assert isinstance(var_dict[name], list)
        for batch_result in var_dict[name]:
            num_channels = batch_result.shape[0]
            mean_abs_diff = (
                (batch_result - running_var[:num_channels]).abs().mean().item()
            )
            var_results[name].append(mean_abs_diff)

    # For each layer, record the mean and std of the average deviations, for
    # both running_mean and running_var.
    ret = {}
    for name, stats in mean_results.items():
        ret[f"{name}_running_mean_MAD_mean"] = torch.tensor(stats).mean().item()
        ret[f"{name}_running_mean_MAD_std"] = torch.tensor(stats).std().item()
    for name, stats in var_results.items():
        ret[f"{name}_running_var_MAD_mean"] = torch.tensor(stats).mean().item()
        ret[f"{name}_running_var_MAD_std"] = torch.tensor(stats).std().item()
    return {"bn_metrics": ret}
