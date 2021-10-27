#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import numpy as np

__all__ = ["cosine_lr"]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def cosine_lr(optimizer, learning_rate, *, warmup_length, epochs):
    def _lr_adjuster(epoch, iteration):
        if epoch < warmup_length:
            lr = _warmup_lr(learning_rate, warmup_length, epoch)
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * learning_rate

        assign_learning_rate(optimizer, lr)
        print(f"Assigned lr={lr}")
        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
