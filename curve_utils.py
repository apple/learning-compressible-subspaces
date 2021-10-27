#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch.nn as nn


def get_stats(model):
    norms = {}
    numerators = {}
    difs = {}
    cossim = 0
    l2 = 0
    num_points = 2

    for i in range(num_points):
        norms[f"{i}"] = 0.0
        for j in range(i + 1, num_points):
            numerators[f"{i}-{j}"] = 0.0
            difs[f"{i}-{j}"] = 0.0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for i in range(num_points):
                vi = get_weight(m, i)
                norms[f"{i}"] += vi.pow(2).sum()
                for j in range(i + 1, num_points):
                    vj = get_weight(m, j)
                    numerators[f"{i}-{j}"] += (vi * vj).sum()
                    difs[f"{i}-{j}"] += (vi - vj).pow(2).sum()

    for i in range(num_points):
        for j in range(i + 1, num_points):
            cossim += numerators[f"{i}-{j}"].pow(2) / (
                norms[f"{i}"] * norms[f"{j}"]
            )
            l2 += difs[f"{i}-{j}"]

    l2 = l2.pow(0.5).item()
    cossim = cossim.item()
    return cossim, l2


def get_weight(m, i):
    if i == 0:
        return m.weight
    return getattr(m, f"weight{i}")


def alpha_bit_map(alpha, **regime_params):
    """
    Maps a continuous alpha value in [0,1] to a bit value. E.g.
    alpha \in [0, 1/8) => 1
    alpha \in [1/8, 2/8) => 2
    alpha \in [2/8, 3/8) => 3
    alpha \in [3/8, 4/8) => 4
    alpha \in [4/8, 5/8) => 5
    alpha \in [5/8, 6/8) => 6
    alpha \in [6/8, 7/8) => 7
    alpha \in [7/8, 8/8) => 8
    """
    min_bits = regime_params["min_bits"]
    max_bits = regime_params["max_bits"]
    distinct_bits = max_bits - min_bits + 1
    for i in range(distinct_bits):
        if i / distinct_bits <= alpha <= (i + 1) / distinct_bits:
            return np.arange(min_bits, max_bits + 1)[i]


def sample_alpha_num_bits(**regime_params):
    discrete = regime_params["discrete"]
    if discrete:
        min_bits = regime_params["min_bits"]
        max_bits = regime_params["max_bits"]
        distinct_bits = max_bits - min_bits + 1
        alpha = (
            np.random.choice(np.arange(1, distinct_bits + 1)) / distinct_bits
        )
    else:
        alpha = np.random.uniform(0, 1)

    num_bits = alpha_bit_map(alpha, **regime_params)

    return alpha, num_bits


def alpha_sampling(**regime_params):
    # biased endpoint sampling
    if regime_params.get(f"alpha_sampling") is not None:
        low, high, endpoint_prob = regime_params.get(f"alpha_sampling")
        if np.random.rand() < endpoint_prob:
            # Pick an endpoint at random.
            if np.random.rand() < 0.5:
                alpha = low
            else:
                alpha = high
        else:
            alpha = np.random.uniform(low, high)
    else:
        alpha = np.random.uniform(0, 1)

    return alpha
