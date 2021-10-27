#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn

import model_logging
import models.networks as networks
import utils
from models.builder import Builder
from models.networks import resprune
from models.networks import utils as network_utils
from models.networks import vggprune
from models.quantized_modules import ConvBn2d
from models.sparse_modules import SparseConv2d


def sparse_module_updates(model, training=False, **regime_params):
    topk = regime_params["topk"]
    current_iteration = regime_params["current_iteration"]
    warmup_iterations = regime_params["warmup_iterations"]
    topk_lower, topk_upper = regime_params["alpha_sampling"][0:2]
    if regime_params["random_param"]:
        if training:
            if current_iteration < warmup_iterations:
                current_topk = topk_upper
            else:
                # Unbiased sampling
                current_topk = np.random.uniform(topk_lower, topk_upper)
        else:
            current_topk = regime_params["topk"]
    else:
        df = np.max([1 - current_iteration / warmup_iterations, 0])
        current_topk = topk + (1 - topk) * df

    for m in model.modules():
        if isinstance(m, SparseConv2d):
            setattr(m, f"topk", current_topk)

    if training:
        regime_params["current_iteration"] += 1

    return model, regime_params


def quantized_module_updates(model, training=False, **regime_params):
    set_bits = regime_params["num_bits"]

    for m in model.modules():
        if isinstance(m, ConvBn2d):
            if regime_params["random_param"]:
                if training:
                    m.start_counter()
                    if m.is_active():
                        bits = np.arange(
                            regime_params["min_bits"],
                            regime_params["max_bits"] + 1,
                        )
                        rand_bits = np.random.choice(bits)
                        m.update_num_bits(rand_bits)
                    else:
                        m.update_num_bits(regime_params["max_bits"])
                else:
                    m.stop_counter()
                    m.update_num_bits(set_bits)
            else:
                m.update_num_bits(set_bits)
                if training:
                    m.start_counter()
                else:
                    m.stop_counter()

    return model, regime_params


def lec_update(model, training=False, **regime_params):
    # The original LEC paper does the update using a global threshold, so we
    # adopt that strategy here.
    model, regime_params = sparse_module_updates(
        model, training=training, **regime_params
    )

    if training:
        return model, regime_params
    else:
        # We create a pruned copy of the model.
        model_kwargs = regime_params["model_kwargs"]
        fresh_copy = utils.make_fresh_copy_of_pruned_network(
            model, model_kwargs
        )

        # The @fresh_copy needs to have its smallest InstanceNorm parameters
        # deleted.
        topk = regime_params["topk"]
        all_weights = []
        for m in fresh_copy.modules():
            if isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
                all_weights.append(m.weight.abs())

        all_weights = torch.cat(all_weights, dim=0)
        y, i = torch.sort(all_weights)
        threshold = y[int(all_weights.shape[0] * (1.0 - topk))]

        for m in fresh_copy.modules():
            if isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
                mask = m.weight.data.clone().abs().gt(threshold).float()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)

        # Now that we have the sparse copy, we slim it down.
        cfg, cfg_mask = network_utils.get_slim_configs(fresh_copy)

        builder = Builder(
            conv_type="StandardConv", bn_type=regime_params["bn_type"]
        )

        try:
            if isinstance(model, nn.DataParallel):
                model = model.module

            if isinstance(model, networks.cpreresnet):
                model_class = resprune
            elif isinstance(model, networks.vgg.vgg):
                model_class = vggprune
            else:
                raise ValueError(
                    "Model {} is not supported for LEC.".format(model)
                )

            _, slimmed_network = model_class.get_slimmed_network(
                fresh_copy.module,
                {"builder": builder, "block_builder": builder, **model_kwargs},
                cfg,
                cfg_mask,
            )
        except IndexError:
            # This is the error if we eliminate a whole layer.
            print(
                f"Something went wrong during LEC - most likely, an entire "
                f"layer was deleted. Using @fresh_copy."
            )
            slimmed_network = fresh_copy
        num_parameters = sum(
            [param.nelement() for param in slimmed_network.parameters()]
        )

        # NOTE: DO NOT use @model here, since it has too many extra buffers.
        total_params = sum(
            [param.nelement() for param in fresh_copy.parameters()]
        )
        regime_params["sparsity"] = (
            total_params - num_parameters
        ) / total_params

        print(f"Got sparsity level of {regime_params['sparsity']}")

        return slimmed_network, regime_params


def ns_update(model, training=False, **regime_params):
    current_width = regime_params["width_factor"]

    for module in model.modules():
        if hasattr(module, "width_factor"):
            module.width_factor = current_width

    return model, regime_params


def us_update(*args, **kwargs):
    return ns_update(*args, **kwargs)


def train(
    model, train_loader, optimizer, criterion, epoch, device, **regime_params
):
    model.zero_grad()
    model.train()
    avg_loss = 0.0

    regime_update_dict = {
        "sparse": sparse_module_updates,
        "quantized": quantized_module_updates,
        "lec": lec_update,
        "ns": ns_update,
        "us": us_update,  # [sic]
    }
    module_update = regime_update_dict[regime_params["regime"]]

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )

        # Special case: you want to do multiple iterations of training.
        if regime_params["regime"] == "ns":
            loss = 0
            optimizer.zero_grad()
            for width_factor in regime_params["width_factors_list"]:
                regime_params["width_factor"] = width_factor

                model, regime_params = module_update(
                    model, True, **regime_params
                )
                output = model(data)
                loss += criterion(output, target)
        elif regime_params["regime"] == "us":
            loss = 0
            optimizer.zero_grad()

            low_factor, high_factor = regime_params["width_factor_limits"]
            n_samples = regime_params["width_factor_samples"]

            assert n_samples >= 2, f"Require n_samples>=2, got {n_samples}"

            width_factors = [low_factor, high_factor]
            for i in range(n_samples - 2):
                width_factors.append(random.uniform(low_factor, high_factor))

            for width_factor in width_factors:
                regime_params["width_factor"] = width_factor

                model, regime_params = module_update(
                    model, True, **regime_params
                )
                output = model(data)
                loss += criterion(output, target)
        else:
            model, regime_params = module_update(model, True, **regime_params)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        if regime_params.get("bn_update_factor") is not None:
            apply_norm_regularization(model, regime_params["bn_update_factor"])
        optimizer.step()

        avg_loss += loss.item()

        log_interval = 10

        if batch_idx % log_interval == 0:
            num_samples = batch_idx * len(data)
            num_epochs = len(train_loader.dataset)
            percent_complete = 100.0 * batch_idx / len(train_loader)

            predicted_labels = output.argmax(dim=1)
            corrects = (
                predicted_labels == target
            ).float().sum() / target.numel()

            print(
                f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f} Correct: {corrects.item():.4f}"
            )

    model.apply(lambda m: setattr(m, "return_feats", False))
    avg_loss = avg_loss / len(train_loader)

    return avg_loss, regime_params


def apply_norm_regularization(model, s_factor):
    count = 0
    for m in model.modules():
        if isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
            count += 1
            m.weight.grad.data.add_(s_factor * torch.sign(m.weight.data))  # L1
    if count == 0:
        raise ValueError(f"Didn't adjust any Norms")


def test(
    model,
    param,
    val_loader,
    criterion,
    epoch,
    device,
    metric_dict=None,
    **regime_params,
):
    # Set eval param
    if regime_params["regime"] in ("sparse", "lec"):
        regime_params["topk"] = param
    elif regime_params["regime"] in ("ns", "us"):
        regime_params["width_factor"] = param
    else:
        regime_params["num_bits"] = param

    metric_dict = _test(
        model,
        param,
        val_loader,
        criterion,
        epoch,
        device,
        metric_dict=metric_dict,
        **regime_params,
    )
    return metric_dict


def _test(
    model,
    param,
    val_loader,
    criterion,
    epoch,
    device,
    metric_dict=None,
    **regime_params,
):

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0

    regime_update_dict = {
        "sparse": sparse_module_updates,
        "quantized": quantized_module_updates,
        "lec": lec_update,
        "ns": ns_update,
        "us": us_update,
    }
    logging_dict = {
        "sparse": model_logging.sparse_logging,
        "quantized": model_logging.quantized_logging,
        "lec": model_logging.lec_logging,
        "ns": model_logging.ns_logging,
        "us": model_logging.us_logging,
    }
    module_update = regime_update_dict[regime_params["regime"]]
    logging = logging_dict[regime_params["regime"]]

    model, regime_params = module_update(model, **regime_params)

    # optionally add the hooks needed for tracking BN accuracy stats.
    if regime_params.get("bn_accuracy_stats", True):
        hooks, mean_dict, var_dict = utils.register_bn_tracking_hooks(model)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):

            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )

            output = model(data)
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    if regime_params.get("bn_accuracy_stats", True):
        utils.unregister_bn_tracking_hooks(hooks)
        extra_metrics = utils.get_bn_accuracy_metrics(
            model, mean_dict, var_dict
        )

        # Mean_dict and var_dict contain a mapping from modules to their
    else:
        extra_metrics = {}

    test_loss /= len(val_loader)
    test_acc = float(correct) / len(val_loader.dataset)
    return logging(
        model,
        test_loss,
        test_acc,
        model_type="indep",
        param=param,
        metric_dict=metric_dict,
        extra_metrics=extra_metrics,
        **regime_params,
    )


def train_model(config):
    # Get network, data, and training parameters
    (
        net,
        opt_params,
        train_params,
        regime_params,
        data,
    ) = utils.network_and_params(config)

    # Unpack training parameters
    epochs, test_freq, _ = train_params

    # Unpack dataset
    trainset, trainloader, testset, testloader = data

    # Unpack optimizer parameters
    criterion, optimizer, scheduler = opt_params

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)

    if device == "cuda":
        cudnn.benchmark = True
    net = torch.nn.DataParallel(net)

    start_epoch = 0

    metric_dict = {"acc": [], "alpha": []}
    if regime_params["regime"] == "quantized":
        metric_dict["num_bits"] = []
    if regime_params["regime"] in (
        "sparse",
        "lec",
        "us",
        "ns",
    ):
        metric_dict["sparsity"] = []

    # Get evaluation parameter grid
    eval_param_grid = regime_params["eval_param_grid"]

    save_dir = regime_params["save_dir"]

    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        scheduler(epoch, None)
        if epoch % test_freq == 0:
            for param in eval_param_grid:
                test(
                    net,
                    param,
                    testloader,
                    criterion,
                    epoch,
                    device,
                    metric_dict=None,
                    **regime_params,
                )

            model_logging.save_model_at_epoch(net, epoch, save_dir)

        _, regime_params = train(
            net,
            trainloader,
            optimizer,
            criterion,
            epoch,
            device,
            **regime_params,
        )

    # Save final model
    for param in eval_param_grid:
        metric_dict = test(
            net,
            param,
            testloader,
            criterion,
            epoch,
            device,
            metric_dict=metric_dict,
            **regime_params,
        )

    model_logging.save_model_at_epoch(net, epoch + 1, save_dir)
    model_logging.save("test_metrics.npy", metric_dict, save_dir)
