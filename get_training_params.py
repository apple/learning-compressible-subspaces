#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import numpy as np

import training_params
import utils


def gen_args_dict(args):
    a_dict = {}

    if args.epochs is not None:
        a_dict["epochs"] = args.epochs

    if args.test_freq is not None:
        a_dict["test_freq"] = args.epochs

    if args.learning_rate is not None:
        a_dict["learning_rate"] = args.learning_rate

    if args.batch_size is not None:
        a_dict["batch_size"] = args.batch_size

    if args.momentum is not None:
        a_dict["momentum"] = args.momentum

    if args.weight_decay is not None:
        a_dict["weight_decay"] = args.weight_decay

    if args.save_dir is not None:
        a_dict["save_dir"] = args.save_dir

    if args.dataset == "imagenet":
        if args.imagenet_dir is not None:
            a_dict["dataset_dir"] = args.imagenet_dir
        else:
            raise ValueError(
                f"ImageNet data directory must be specified as --imagenet_dir <dir>"
            )

    return a_dict


def unstructured_args_dict(args):
    ua_dict = gen_args_dict(args)

    if args.topk is not None:
        ua_dict["topk"] = args.topk

    if args.warmup_budget is not None:
        ua_dict["warmup_budget"] = args.warmup_budget

    if args.eval_topk_grid is not None:
        if args.method == "lcs_l":
            ua_dict["alpha_grid"] = args.eval_topk_grid
        else:
            ua_dict["eval_param_grid"] = args.eval_topk_grid

    if (
        args.topk_lower_bound is not None
        and args.topk_upper_bound is not None
        and args.eval_topk_grid is not None
    ):
        if args.method == "lcs_l":
            ua_dict["alpha_grid"] = args.eval_topk_grid
        elif args.method == "lcs_p":
            ua_dict["eval_param_grid"] = args.eval_topk_grid

        ua_dict["alpha_sampling"] = [
            args.topk_lower_bound,
            args.topk_upper_bound,
            0.5,
        ]

    return ua_dict


def structured_args_dict(args, base_config):
    sa_dict = gen_args_dict(args)

    if args.width_factors_list is not None and args.method == "ns":
        builder_kwargs = base_config["parameters"]["builder_kwargs"]
        builder_kwargs["width_factors_list"] = args.width_factors_list
        sa_dict["builder_kwargs"] = builder_kwargs

    if args.width_factor_limits is not None and args.method != "ns":
        regime_params = base_config["parameters"]["regime_params"]
        regime_params["width_factor_limits"] = args.width_factor_limits
        sa_dict["regime_params"] = regime_params

    if args.width_factor_samples is not None and args.method != "ns":
        regime_params = sa_dict.get(
            "regime_params", base_config["parameters"]["regime_params"]
        )
        regime_params["width_factor_samples"] = args.width_factor_samples
        sa_dict["regime_params"] = regime_params

    if args.eval_width_factors is not None:
        regime_params = sa_dict.get(
            "regime_params", base_config["parameters"]["regime_params"]
        )
        if args.method in ("us", "ns", "lcs_p"):
            regime_params["eval_width_factors"] = args.eval_width_factors
            sa_dict["regime_params"] = regime_params
            sa_dict["eval_param_grid"] = args.eval_width_factors
        elif args.method == "lcs_l":
            w_l, w_u = regime_params["width_factor_limits"]
            alpha_grid = [
                (w_f - w_l) / (w_u - w_l) for w_f in args.eval_width_factors
            ]
            sa_dict["alpha_grid"] = alpha_grid

    return sa_dict


def quantized_args_dict(args):
    q_dict = gen_args_dict(args)

    if args.bit_width is not None and args.method == "target_bit_width":
        q_dict["num_bits"] = args.bit_width

    if args.eval_bit_widths is not None and args.method != "lcs_l":
        q_dict["eval_param_grid"] = args.eval_bit_widths

    if args.bit_width_limits is not None and args.method in ("lcs_p", "lcs_l"):
        min_bits, max_bits = [int(x) for x in args.bit_width_limits]
        q_dict["min_bits"] = min_bits
        q_dict["max_bits"] = max_bits
        if args.method == "lcs_l":
            range_len = max_bits - min_bits + 1
            alpha_grid = [
                np.floor(x * 1000) / 1000
                for x in np.linspace(0, 1, range_len + 1)
            ][1:]
            q_dict["alpha_grid"] = alpha_grid
        elif args.method == "lcs_p":
            if args.eval_bit_widths is None:
                eval_param_grid = np.arange(min_bits, max_bits + 1)
                q_dict["eval_param_grid"] = eval_param_grid

    return q_dict


def get_config_norm(params):
    norm_types = ["IN", "BN", "GN"]
    base_bn_type = params["bn_type"]
    base_block_bn_type = params["block_bn_type"]
    for n in norm_types:
        if n in base_bn_type:
            base_bn = n
        if n in base_block_bn_type:
            base_block_bn = n

    for norm in norm_types:
        if norm in base_bn and norm in base_block_bn:
            return norm


def get_method_config(args, setting):
    method = args.method

    base_config_dir = f"configs/{setting}/{method}.yaml"

    try:
        base_config = utils.get_yaml_config(base_config_dir)
    except:
        raise ValueError(f"{setting}/{method} not valid training configuration")

    params = base_config["parameters"]

    # Update base config with specific parameters

    # Set normalization layers
    config_norm = get_config_norm(params)
    norm_to_use = config_norm if args.norm is None else args.norm
    norm_types = ["IN", "BN", "GN"]
    if norm_to_use not in norm_types:
        raise ValueError(
            f"Norm {norm_to_use} not valid. Supported normalization types: IN, BN, GN."
        )

    base_bn_type = base_config["parameters"]["bn_type"]
    base_block_bn_type = base_config["parameters"]["block_bn_type"]
    for n in norm_types:
        if n in base_bn_type:
            base_bn = n
        if n in base_block_bn_type:
            base_block_bn = n

    if setting == "quantized" and norm_to_use == "BN":
        base_config["parameters"]["bn_type"] = "QuantStandardBN"
        base_config["parameters"]["block_bn_type"] = "QuantStandardBN"
    else:
        base_config["parameters"]["bn_type"] = base_bn_type.replace(
            base_bn, norm_to_use
        )
        base_config["parameters"]["block_bn_type"] = base_block_bn_type.replace(
            base_block_bn, norm_to_use
        )
        # Set track_running_stats to True if using BN
        if norm_to_use == "BN":
            base_norm_kwargs = base_config["parameters"].get(
                "norm_kwargs", None
            )
            if base_norm_kwargs is not None:
                base_norm_kwargs["track_running_stats"] = True
            else:
                base_config["parameters"]["norm_kwargs"] = {
                    "track_running_stats": True
                }

    # Update dataset
    dataset = args.dataset
    base_config["parameters"]["dataset"] = dataset

    # Update model
    model_name = args.model.lower()
    base_config["parameters"]["model_config"]["model_class"] = model_name
    model_kwargs = params.get("model_kwargs", None)
    if model_kwargs is not None:
        base_model_kwargs = base_config["parameters"]["model_config"].get(
            "model_kwargs", None
        )
        if base_model_kwargs is not None:
            base_model_kwargs.update(model_kwargs)
        else:
            base_config["parameters"]["model_config"]["model_kwargs"] = {}

    # Remove channel_selection_active for models without this parameter
    if model_name not in ("cpreresnet20"):
        base_model_kwargs = base_config["parameters"]["model_config"][
            "model_kwargs"
        ]
        base_model_kwargs.pop("channel_selection_active", None)

    # For unstructured sparsity, if using GN, set num_groups to 32
    # We cannot use GN with structured sparsity (number of channels isn't always
    # divisible by 32), we use IN instead.
    if norm_to_use == "GN":
        if setting in ("unstructured_sparsity", "quantized"):
            num_groups = 32
        else:
            raise NotImplementedError(
                f"GroupNorm disabled for setting={setting}."
            )
        base_norm_kwargs = base_config["parameters"].get("norm_kwargs", None)
        if base_norm_kwargs is not None:
            base_norm_kwargs["num_groups"] = num_groups
        else:
            base_config["parameters"]["norm_kwargs"] = {
                "num_groups": num_groups
            }

    # Set default model training parameters
    model_training_params = training_params.model_data_params(args)
    base_config["parameters"].update(model_training_params)

    # Update training parameters with user-specified ones
    if setting == "unstructured_sparsity":
        args_dict = unstructured_args_dict(args)
    elif setting == "structured_sparsity":
        args_dict = structured_args_dict(args, base_config)
    elif setting == "quantized":
        args_dict = quantized_args_dict(args)
    else:
        args_dict = {}

    base_config["parameters"].update(args_dict)

    # Get/make save directory
    args_save_dir = args.save_dir
    if args_save_dir is None:
        config_save_dir = params["save_dir"]
        save_dir = utils.create_save_dir(
            config_save_dir, method, model_name, dataset, norm_to_use
        )
    else:
        save_dir = utils.create_save_dir(
            args_save_dir, method, model_name, dataset, norm_to_use, False
        )

    base_config["parameters"]["save_dir"] = save_dir

    # Print training details
    utils.print_train_params(
        base_config, setting, method, norm_to_use, save_dir
    )

    return base_config
