#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
"""Utility functions to tag tensors with metadata.

The metadata remains with the tensor under torch operations that don't change
the values, e.g. .clone(), .contiguous(), .permute(), etc.
"""

import collections
import copy
from typing import Any
from typing import Optional

import numpy as np
import torch

QuantizeAffineParams2 = collections.namedtuple(
    "QuantizeAffineParams", ["scale", "zero_point", "num_bits"]
)


class _SpecialTensor(torch.Tensor):
    """This class denotes special tensors.

    It isn't intended to be used directly, but serves as a helper for tagging
    tensors with metadata.

    It subclasses torch.Tensor so that isinstance(t, torch.Tensor) returns True
    for special tensors. It forbids some of the methods of torch.Tensor, and
    overrides a few methods used to create other tensors, to ensure the result
    is still special.
    """

    _metadata = None

    def __getattribute__(self, attr: str) -> Any:
        # Disallow new_zeros, new_ones, new_full, etc.
        if "new_" in attr:
            raise AttributeError(
                "Invalid attr {!r} for special tensors".format(attr)
            )
        return super().__getattribute__(attr)

    def detach(self) -> "_SpecialTensor":
        ret = super().detach()
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    @property
    def data(self) -> "_SpecialTensor":
        ret = super().data
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def clone(self) -> "_SpecialTensor":
        ret = super().clone()
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def cuda(
        self, device: Optional[torch.device] = None, non_blocking: bool = False
    ) -> "_SpecialTensor":
        ret = super().cuda()
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def contiguous(self) -> "_SpecialTensor":
        ret = super().contiguous()
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def view(self, *args, **kwargs) -> "_SpecialTensor":
        ret = super().view(*args, **kwargs)
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def permute(self, *args, **kwargs) -> "_SpecialTensor":
        ret = super().permute(*args, **kwargs)
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def __getitem__(self, *args, **kwargs) -> "_SpecialTensor":
        ret = super().__getitem__(*args, **kwargs)
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret

    def __copy__(self) -> "_SpecialTensor":
        ret = copy.copy(super())
        ret.__class__ = _SpecialTensor
        ret._metadata = self._metadata
        return ret


def _check_type(tensor: torch.Tensor) -> None:
    given_type = type(tensor)
    if not issubclass(given_type, torch.Tensor):
        raise TypeError("invalid type {!r}".format(given_type))


def tag_with_metadata(tensor: torch.Tensor, metadata: Any) -> None:
    """Tag a metadata to a tensor."""
    _check_type(tensor)
    tensor.__class__ = _SpecialTensor
    tensor._metadata = metadata


RepresentibleByQuantizeAffine = collections.namedtuple(
    "RepresentibleByQuantizeAffine", ["quant_params"]
)


def mark_quantize_affine(
    tensor: torch.Tensor,
    scale: float,
    zero_point: int,
    dtype: np.dtype = np.uint8,
) -> None:
    """Mark a tensor as quantized with affine.

    See //xnorai/training/pytorch/extensions/functions:quantize_affine for more
    info on this method of quantization.

    The tensor itself can be a floating point Tensor. However, its values must
    be representible with @scale and @zero_point. This function, for performance
    reasons, does not validiate if the tensor is really quantizable as it
    claims to be.

    Arguments:
        tensor (torch.Tensor): The tensor to be marked as affine-quantizable
            Tensor.
        scale (float): the scale (from quantization parameters).
        zero_point (int): The zero_point (from quantization parameters).
        dtype (numpy.dtype): Type of tensor when quantized (this is usually
            numpy.uint8, which is used for Q8). A ValueError will be thrown if
            the input dtype is not one of the following:
                {numpy.uint8, numpy.int32}.
    """
    allowed_dtypes = [np.uint8, np.int32]
    if dtype not in allowed_dtypes:
        raise ValueError(
            "Provided dtype ({}) is not supported. Please use: {}".format(
                dtype, allowed_dtypes
            )
        )
    quant_params = QuantizeAffineParams2(scale, zero_point, dtype)
    tag_with_metadata(tensor, RepresentibleByQuantizeAffine(quant_params))
