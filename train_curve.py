#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

# Trains a standard/sparse/quantized line/poly chain/bezier curve.

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import curve_utils
import model_logging
import models.networks as models
import utils
from models import modules
from models.builder import Builder
from models.networks import resprune
from models.networks import utils as network_utils
from models.networks import vggprune
from models.quantized_modules import ConvBn2d


def sparse_module_updates(model, training=False, alpha=None, **regime_params):
    current_iteration = regime_params["current_iteration"]
    warmup_iterations = regime_params["warmup_iterations"]

    if alpha is None:
        alpha = curve_utils.alpha_sampling(**regime_params)

    df = np.max([1 - current_iteration / warmup_iterations, 0])
    topk = alpha + (1 - alpha) * df

    is_standard = regime_params["is_standard"]

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(
            m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)
        ):
            setattr(m, f"alpha", alpha)

            if (not is_standard) and hasattr(m, "topk"):
                setattr(m, "topk", topk)

    if training:
        regime_params["current_iteration"] += 1

    return model, regime_params


def quantized_module_updates(
    model, training=False, alpha=None, **regime_params
):
    if alpha is None:
        alpha, num_bits = curve_utils.sample_alpha_num_bits(**regime_params)
    else:
        num_bits = curve_utils.alpha_bit_map(alpha, **regime_params)

    is_standard = regime_params["is_standard"]

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(
            m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)
        ):
            setattr(m, f"alpha", alpha)

            if isinstance(m, ConvBn2d):
                if not is_standard:
                    m.update_num_bits(num_bits)

                if training:
                    m.start_counter()
                else:
                    m.stop_counter()

    return model, regime_params


def _test_time_lec_update(model, **regime_params):
    # This requires that the topk values are already set on the model.

    # We create a whole new copy of the model which is pruned.
    model_kwargs = regime_params["model_kwargs"]
    fresh_copy = utils.make_fresh_copy_of_pruned_network(model, model_kwargs)
    cfg, cfg_mask = network_utils.get_slim_configs(fresh_copy)

    builder = Builder(conv_type="StandardConv", bn_type="StandardIN")

    try:
        if isinstance(model, models.cpreresnet):
            model_class = resprune
        elif isinstance(model, models.vgg.vgg):
            model_class = vggprune
        else:
            raise ValueError(
                "Model {} is not surpported for LEC.".format(model)
            )

        _, slimmed_network = model_class.get_slimmed_network(
            fresh_copy.module,
            {"builder": builder, "block_builder": builder, **model_kwargs},
            cfg,
            cfg_mask,
        )
    except:
        print(
            f"Something went wrong during LEC. Most likely, an entire "
            f"layer was deleted. Using @fresh_copy."
        )
        slimmed_network = fresh_copy
    num_parameters = sum(
        [param.nelement() for param in slimmed_network.parameters()]
    )

    # NOTE: DO NOT use @model here, since it has too many extra buffers in the
    # case of training a line.
    total_params = sum([param.nelement() for param in fresh_copy.parameters()])
    regime_params["sparsity"] = (total_params - num_parameters) / total_params

    print(f"Got sparsity level of {regime_params['sparsity']}")

    return slimmed_network, regime_params


def lec_update(model, training=False, alpha=None, **regime_params):
    # Note: this file is for training a *curve*, so we do the right thing for
    # curves here. Namely, we apply the same update to alpha/topk as when
    # training a standard curve, but we gather the sparsity in a different way
    # since we're doing LEC rather than doing the unstructured sparsity.
    model, regime_params = sparse_module_updates(
        model, training=training, alpha=alpha, **regime_params
    )

    if training:
        return model, regime_params
    else:
        return _test_time_lec_update(model, **regime_params)


def us_update(model, training=False, alpha=None, **regime_params):
    assert alpha is not None, f"Alpha value is required."

    # Use alpha to compute the width factor.
    low_factor, high_factor = regime_params["width_factor_limits"]
    width_factor = low_factor + (high_factor - low_factor) * alpha
    regime_params["width_factor"] = width_factor

    for module in model.modules():
        if hasattr(module, "width_factor"):
            module.width_factor = width_factor

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(
            m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)
        ):
            setattr(m, f"alpha", alpha)

    return model, regime_params


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
        "us": us_update,
    }
    module_update = regime_update_dict[regime_params["regime"]]

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        if regime_params["regime"] == "us":
            loss = 0

            if regime_params["width_factor_sampling_method"] == "sandwich":
                n_samples = regime_params["width_factor_samples"]

                assert n_samples >= 2, f"Require n_samples>=2, got {n_samples}"

                alphas = [0, 1]
                for i in range(n_samples - 2):
                    alphas.append(random.uniform(0.0, 1.0))
            elif regime_params["width_factor_sampling_method"] == "point":
                alphas = [random.uniform(0.0, 1.0)]
            else:
                raise NotImplementedError

            for alpha in alphas:
                model, regime_params = module_update(
                    model, True, alpha=alpha, **regime_params
                )
                output = model(data)
                loss += criterion(output, target)

        else:
            model, regime_params = module_update(
                model, training=True, **regime_params
            )
            output = model(data)
            loss = criterion(output, target)

        # Application of the regularization term, equation 3.
        num_points = regime_params["num_points"]
        beta = regime_params.get("beta", 1)
        if beta > 0 and num_points > 1:
            out = random.sample([i for i in range(num_points)], 2)

            i, j = out[0], out[1]
            num = 0.0
            normi = 0.0
            normj = 0.0
            for m in model.modules():
                # Apply beta term if we have a conv, and (optionally) if we have
                # a norm layer. Only apply beta term if alpha exists (e.g. it's
                # a line).
                # (We forbid an exact type match because "plain-old" Conv2d
                # layers [as in LEC] should not trigger this logic).
                should_apply_beta = isinstance(m, nn.Conv2d) and not type(
                    m
                ) in (nn.Conv2d, modules.AdaptiveConv2d)
                should_apply_beta = should_apply_beta or (
                    isinstance(
                        m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)
                    )
                    and regime_params.get("apply_beta_to_norm", False)
                )
                should_apply_beta = should_apply_beta and hasattr(m, "alpha")
                if should_apply_beta:
                    vi = curve_utils.get_weight(m, i)
                    vj = curve_utils.get_weight(m, j)
                    num += (vi * vj).sum()
                    normi += vi.pow(2).sum()
                    normj += vj.pow(2).sum()
            loss += beta * (num.pow(2) / (normi * normj))

        loss.backward()
        if regime_params.get("bn_update_factor") is not None:
            loss_reg_term = (
                utils.apply_in_topk_reg(model, apply_to_bias=False)
                * regime_params["bn_update_factor"]
            )
        else:
            loss_reg_term = 0
        optimizer.step()

        avg_loss += loss.item() + loss_reg_term

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


def test(
    model,
    alpha,
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
        "us": us_update,
    }
    logging_dict = {
        "sparse": model_logging.sparse_logging,
        "quantized": model_logging.quantized_logging,
        "lec": model_logging.lec_logging,
        "us": model_logging.us_logging,
    }
    module_update = regime_update_dict[regime_params["regime"]]
    logging = logging_dict[regime_params["regime"]]

    model, regime_params = module_update(model, alpha=alpha, **regime_params)

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
        model_type="curve",
        param=alpha,
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
    epochs, test_freq, alpha_grid = train_params

    # Unpack dataset
    trainset, trainloader, testset, testloader = data

    # Unpack optimization parameters
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

    save_dir = regime_params["save_dir"]

    for epoch in range(start_epoch, start_epoch + epochs):
        scheduler(epoch, None)
        if epoch % test_freq == 0:
            for alpha in alpha_grid:
                test(
                    net,
                    alpha,
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
    for alpha in alpha_grid:
        metric_dict = test(
            net,
            alpha,
            testloader,
            criterion,
            epoch,
            device,
            metric_dict=metric_dict,
            **regime_params,
        )

    model_logging.save_model_at_epoch(net, epoch + 1, save_dir)
    model_logging.save("test_metrics.npy", metric_dict, save_dir)
