#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn

import utils


class SparseConv2d(nn.Conv2d):
    """
    A class for sparse 2d convolution layers
    """

    def __init__(self, *args, **kwargs):
        """
        Inputs:
        - topk: A float between 0 and 1 indicating the percentage of top values
              to keep. E.g. if topk = 0.3, will keep the top 30% of weights (by
               magnitude) and drop the rest
        """
        self.method = kwargs.pop("method", "topk")
        self.topk = 1.0
        super(SparseConv2d, self).__init__(*args, **kwargs)

    def apply_sparsity(self, weight: torch.Tensor) -> torch.Tensor:
        return utils.apply_sparsity(self.topk, weight, method=self.method)

    def forward(self, x, weight=None):
        """
        Performs forward pass after passing weight tensor through apply_sparsity. Iterations are incremented only on train
        """
        if weight is None:
            return self._conv_forward(
                x, self.apply_sparsity(self.weight), self.bias
            )
        else:
            return self._conv_forward(x, self.apply_sparsity(weight), self.bias)

    def __repr__(self) -> str:
        ret = super().__repr__()
        # Remove last paren.
        ret = ret[:-1]
        ret += f", method={self.method})"
        return ret


class SparseSubspaceConv(SparseConv2d):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        return super().forward(x, w)


class SparseTwoParamConv(SparseSubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight)
        initialize_fn(self.weight1)


class SparseLinesConv(SparseTwoParamConv):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w
