#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import train_curve
import train_indep
from get_training_params import get_method_config


def train(args, setting):
    config = get_method_config(args, setting)
    if config["parameters"]["script"] == "train_curve.py":
        train_curve.train_model(config)
    else:
        train_indep.train_model(config)
