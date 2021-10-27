#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
# Architecture parameters

DEFAULT_PARAMS = {
    "epochs": 200,
    "test_freq": 20,
    "batch_size": 128,
    "momentum": 0.9,
    "weight_decay": 0.0005,
}

DEFAULT_IMAGENET_PARAMS = {
    **DEFAULT_PARAMS,
    "epochs": 90,
    "test_freq": 1,
    "batch_size": 128,
    "weight_decay": 0.00005,
}

VGG_IMAGENET_PARAMS = {
    **DEFAULT_IMAGENET_PARAMS,
    "batch_size": 256,
}


def model_data_params(args):
    model = args.model
    dataset = args.dataset

    if dataset == "imagenet":
        if model == "vgg19":
            return VGG_IMAGENET_PARAMS
        elif model == "resnet18":
            return DEFAULT_IMAGENET_PARAMS
        else:
            raise NotImplementedError(
                f"No training parameters for {model}/{dataset}"
            )
    elif "cifar" in dataset:
        return DEFAULT_PARAMS
    else:
        raise NotImplementedError(
            f"No training parameters for {model}/{dataset}"
        )
