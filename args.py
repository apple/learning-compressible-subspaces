#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import argparse


def gen_args(desc):
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--model",
        type=str,
        default="cpreresnet20",
        help="Which model architecture to use. One of crepresnet20, resnet18, vgg19",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Which dataset to use. One of cifar10, imagenet",
    )

    parser.add_argument(
        "--imagenet_dir",
        type=str,
        help="The root directory to the ImageNet dataset",
    )

    parser.add_argument("--save_dir", type=str, help="Directory to save model")

    parser.add_argument(
        "--norm",
        type=str,
        help="Layer normalization type. One of BN, IN, GN",
    )

    parser.add_argument("--epochs", type=int, help="Number of training epochs")

    parser.add_argument(
        "--test_freq", type=int, help="Number of epochs between testing"
    )

    parser.add_argument(
        "--learning_rate", type=float, help="Optimizer learning rate"
    )

    parser.add_argument(
        "--batch_size", type=int, help="Training/test batch size"
    )

    parser.add_argument("--momentum", type=float, help="Optimizer momentum")

    parser.add_argument(
        "--weight_decay", type=float, help="L2 regularization parameter"
    )

    return parser


def parse_unstructured_arguments():
    parser = gen_args("Unstructured Sparsity Training")

    parser.add_argument(
        "--method",
        type=str,
        default="lcs_l",
        help="Training method. One of topk_target, lcs_p, lcs_l",
    )

    parser.add_argument(
        "--topk",
        type=float,
        help="Target topk (0,1) for BN training",
    )

    parser.add_argument(
        "--topk_lower_bound",
        type=float,
        help="Lower bound (high accuracy) endpoint of the learned subspace",
    )

    parser.add_argument(
        "--topk_upper_bound",
        type=float,
        help="Upper bound (high efficiency) endpoint of the learned subspace",
    )

    parser.add_argument(
        "--warmup_budget",
        type=float,
        help="Value in range (0,100] denoting the percentage of epochs for warmup phase",
    )

    parser.add_argument(
        "--eval_topk_grid",
        type=lambda x: [float(w) for w in x.split(",")],
        help="Will evaluate at these topk values",
    )

    return parser.parse_args()


def parse_structured_arguments():
    parser = gen_args("Structured Sparsity Training")

    parser.add_argument(
        "--method",
        type=str,
        default="lcs_l",
        help="Training method. One of lec, ns, us, lcs_l, lcs_p",
    )

    parser.add_argument(
        "--width_factors_list",
        type=lambda x: [float(w) for w in x.split(",")],
        help="Desired width factors for NS. Ex: --width_factors_list 0.25,0.5,0.75,1.0",
    )

    parser.add_argument(
        "--width_factor_limits",
        type=lambda x: [float(w) for w in x.split(",")],
        help="US width factor lower and upper bounds. Ex: --width_factor_limits 0.25,1.0",
    )

    parser.add_argument(
        "--width_factor_samples",
        type=int,
        help="Number of width factor samples for US sandwich rule",
    )

    parser.add_argument(
        "--eval_width_factors",
        "--list",
        type=lambda x: [float(w) for w in x.split(",")],
        help="Width factors at which to evaluate model. Ex: --eval_width_factors 0.25,0.5,0.75,1.0",
    )

    return parser.parse_args()


def parse_quantized_arguments():
    parser = gen_args("Quantized Training")

    parser.add_argument(
        "--method",
        type=str,
        default="lcs_l",
        help="Training method. One of target_bit_width, lcs_p, lcs_l",
    )

    parser.add_argument(
        "--bit_width",
        type=int,
        help="Target bit width after warmup phase. Used for target_bit_width training",
    )

    parser.add_argument(
        "--eval_bit_widths",
        type=lambda x: [float(w) for w in x.split(",")],
        help="Number of bits at which to evaluate. Ex: --eval_num_bits 3,4,5",
    )

    parser.add_argument(
        "--bit_width_limits",
        type=lambda x: [float(w) for w in x.split(",")],
        help="Min/max number of bits to train line. Ex: --bit_width_limits 3,8",
    )

    return parser.parse_args()


def validate_model_data(args):
    """
    Ensures that the specified model and dataset are compatible
    """
    implemented_models = ["cpreresnet20", "resnet18", "vgg19"]
    implemented_datasets = ["cifar10", "imagenet"]

    if args.model not in implemented_models:
        raise ValueError(f"{args.model} not implemented.")

    if args.dataset not in implemented_datasets:
        raise ValueError(f"{args.dataset} not implemented.")

    if args.dataset == "imagenet":
        if args.model not in ("resnet18", "vgg19"):
            raise ValueError(
                f"{args.model} does not support ImageNet. Supported models: resnet18, vgg19"
            )
    elif args.dataset == "cifar10":
        if args.model not in ("cpreresnet20"):
            raise ValueError(
                f"{args.model} does not support CIFAR. Supported models: cpreresnet20"
            )

    return args


def validate_unstructured_params(args):
    """
    Esnures that the specified unstructured sparsity parameters are valid
    """
    lb = args.topk_lower_bound
    ub = args.topk_upper_bound

    if (lb is not None and ub is None) or (lb is None and ub is not None):
        raise ValueError("Both upper and lower TopK bounds must be specified")

    if lb is not None:
        if lb < 0:
            raise ValueError("TopK lower bound must be >= 0")
        if ub > 1:
            raise ValueError("TopK upper bound must be <= 1")
        if lb >= ub:
            raise ValueError("TopK lower bound must be < upper bound")
        if args.eval_topk_grid is None:
            raise ValueError(
                "eval_topk_grid must be specified when TopK bounds are"
            )

    return args


def validate_structured_params(args):
    """
    Ensures that the specified structured sparsity parameters are valid
    """
    if args.method in ("lec", "ns"):
        if args.norm is not None and args.norm != "BN":
            raise ValueError("LEC and NS only implemented for BN layers")

        if args.method == "lec" and args.model not in ("cpreresnet20", "vgg19"):
            raise ValueError("LEC only implemented for cpreresnet20, vgg19")

    return args


def validate_quant_params(args):
    """
    Ensures that the specified quantized parameters are valid
    """
    bit_range = [3, 4, 5, 6, 7, 8]

    if args.method == "target_bit_width":
        if args.bit_width is not None and args.bit_width not in bit_range:
            raise ValueError("Bit width must be one of 3, 4, 5, 6, 7, 8.")

    if args.bit_width_limits is not None:
        l_bit, u_bit = args.bit_width_limits
        if l_bit < 3:
            raise ValueError("Smallest bit width must be >= 3")

        if u_bit > 8:
            raise ValueError("Largest bit width must be <= 8")

    return args


def unstructured_args():
    args = parse_unstructured_arguments()
    args = validate_unstructured_params(args)

    return validate_model_data(args)


def structured_args():
    args = parse_structured_arguments()
    args = validate_structured_params(args)

    return validate_model_data(args)


def quantized_args():
    args = parse_quantized_arguments()
    args = validate_quant_params(args)

    return validate_model_data(args)
