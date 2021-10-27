#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn


def get_slim_configs(model):
    # Note: the model should *already* be pruned on the BN layers. This module
    # will not apply the pruning part.
    # We expect the user to create a (pruned) copy of the model first before
    # calling this.
    is_cuda = next(model.parameters()).is_cuda

    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
            mask = m.weight.abs() > 0
            if is_cuda:
                mask = mask.cuda()
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print(
                "layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}".format(
                    k, mask.shape[0], int(torch.sum(mask))
                )
            )
        elif isinstance(m, nn.MaxPool2d):
            cfg.append("M")

    return cfg, cfg_mask
