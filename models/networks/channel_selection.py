#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
# Only used for Learning Efficient Convolutions (LEC) experiments.
# Deactivated in other cases.

import numpy as np
import torch
import torch.nn as nn


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put
    directly after BatchNorm2d layer.

    The output shape of this layer is determined by the number of 1s in
    `self.indexes`.
    """

    def __init__(self, num_channels, active=True):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))
        self.active = active

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        if not self.active:
            return input_tensor
        selected_index = np.squeeze(
            np.argwhere(self.indexes.data.cpu().numpy())
        )
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(active={self.active})"
