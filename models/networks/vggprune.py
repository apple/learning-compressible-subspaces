#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn

from .vgg import *


def get_slimmed_network(model, model_kwargs, cfg, cfg_mask):
    assert not isinstance(model, nn.DataParallel), f"Must unwrap DataParallel"

    is_cuda = next(model.parameters()).is_cuda
    print("Cfg:")
    print(cfg)

    newmodel = vgg(cfg=cfg, **model_kwargs)

    if is_cuda:
        newmodel.cuda()

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]

        if isinstance(m0, (nn.modules.batchnorm._NormBase, nn.GroupNorm)):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d) and m0 != model.classifier:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print(
                "In shape: {:d}, Out shape {:d}.".format(idx0.size, idx1.size)
            )
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif m0 == model.classifier:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()

            assert m1.bias is None == m0.bias is None
            if m1.bias is not None:
                m1.bias.data = m0.bias.data.clone()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])

    return num_parameters, newmodel
