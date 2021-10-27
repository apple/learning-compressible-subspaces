#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import collections
import numbers
from typing import Any
from typing import Optional

import numpy as np
import torch
from torch import autograd
from torch import nn

from .special_tensors import RepresentibleByQuantizeAffine
from .special_tensors import tag_with_metadata

QuantizeAffineParams2 = collections.namedtuple(
    "QuantizeAffineParams", ["scale", "zero_point", "num_bits"]
)

INFINITY = 1e10


def _validate_tensor(tensor: torch.Tensor) -> None:
    if torch.isnan(tensor).any():
        raise ValueError("Found NaN in the tensor.")
    if tensor.abs().max() > INFINITY:
        raise ValueError(
            "Tensor seems to be diverging. Found a value > {}".format(INFINITY)
        )


def get_quantized_representation(
    tensor: torch.Tensor,
    quantize_params: QuantizeAffineParams2,
) -> torch.Tensor:
    """Gets the quantize representation of a float @tensor.

    The resulting tensor will contain the quantized values and the quantization
    parameters will be tagged with the tensor as a special tensor.

    A ValueError will be raised if the given tensor contains NaN or divergent
    values.

    Arguments:
        tensor (torch.Tensor): The float torch tensor to quantize.
        quantize_params (QuantizeAffineParams): The quantization params to
            quantize the tensor by.
    """
    _validate_tensor(tensor)

    scale = quantize_params.scale
    zero_point = quantize_params.zero_point
    num_bits = quantize_params.num_bits
    if scale == 0:
        # Special case, all elements are zeros.
        if zero_point != 0:
            raise ValueError(
                "The given QuantizeAffineParams (={}) has a non-zero zero point"
                " with a scale of 0.".format(quantize_params)
            )
        quantized_tensor = torch.zeros_like(tensor)
        tag_with_metadata(quantized_tensor, quantize_params)
        return quantized_tensor

    qmin, qmax = get_qmin_qmax(num_bits)
    reciprocal = 1 / scale
    quantized_tensor = ((tensor * reciprocal).round_() + zero_point).clamp_(
        qmin, qmax
    )

    tag_with_metadata(quantized_tensor, quantize_params)
    return quantized_tensor


def mark_quantize_affine(
    tensor: torch.Tensor,
    scale: float,
    zero_point: int,
    dtype: np.dtype = np.uint8,
) -> None:
    """Mark a tensor as quantized with affine.

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
    quant_params = QuantizeAffineParams2(scale, zero_point, dtype)
    tag_with_metadata(tensor, RepresentibleByQuantizeAffine(quant_params))


class QuantizeAffineFunction(autograd.Function):
    """Simulates affect of affine quantization during forward pass.

    This function simulates the affect of quantization and subsequent
    dequantization (in the forward pass only). Although the affine
    transformation results in a different basis (e.g. uint8), the output of this
    function will be a float Tensor representing that transformation (the
    dequantized Tensor).

    A ValueError will be raised if the input or resulting tensor contains NaN or
    divergent values.

    Arguments:
        input (Tensor): The input float Tensor to be quantized.
        quantize_params (quantize_affine_util.QuantizeAffineParams): The
            quantization parameter to quantize the input tensor by.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        quantize_params: QuantizeAffineParams2,
    ) -> torch.Tensor:
        quantized_tensor = get_quantized_representation(input, quantize_params)
        dequantized_tensor = dequantize(quantized_tensor, quantize_params)

        mark_quantize_affine(
            dequantized_tensor,
            quantize_params.scale,
            quantize_params.zero_point,
            quantize_params.num_bits,
        )
        return dequantized_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        """We will approximate the gradient as the identity"""
        return grad_output, None


def quantize_affine_function_continuous(
    input: torch.Tensor,
    quantize_params: QuantizeAffineParams2,
) -> torch.Tensor:
    quantized_tensor = get_quantized_representation(input, quantize_params)
    dequantized_tensor = dequantize(quantized_tensor, quantize_params)

    mark_quantize_affine(
        dequantized_tensor,
        quantize_params.scale,
        quantize_params.zero_point,
        quantize_params.num_bits,
    )
    return dequantized_tensor


def get_qmin_qmax(num_bits):
    return -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1


def get_quantization_params(
    rmin: float,
    rmax: float,
    num_bits: int = 8,
) -> QuantizeAffineParams2:
    """Returns QuantizeAffineParams for a data range [rmin, rmax].

    The range must include 0 otherwise that's a failure. The scale and
    zero_point are picked such that the error is quantization error is
    minimized.

    Arguments:
        rmin (float): The data minimum point. Numbers smaller than rmin would
            not be representible by the quantized schema.
        rmax (float): The data maximum point. Numbers bigger than rmax would
            not be representible by the quantized schema.
        dtype (optional, np.dtype): The dtype that should be used to represent
            the individual numbers after quantization. Only np.uint8 is
            supported.
    """
    if rmin > rmax:
        raise ValueError("Got rmin (={}) > rmax (={}).".format(rmin, rmax))
    if rmin > 0 or rmax < 0:
        raise ValueError(
            "The data range ([{}, {}]) must always include "
            "0.".format(rmin, rmax)
        )

    if rmin == rmax == 0.0:
        # Special case: all values are zero.
        return QuantizeAffineParams2(0, 0, num_bits)

    # Scale is floating point and is (rmax - rmin) / (qmax - qmin) to map the
    # length of the ranges. For zero_point, we solve the following equation:
    #       rmin = (qmin - zero_point) * scale
    qmin, qmax = get_qmin_qmax(num_bits)
    scale = (rmax - rmin) / (qmax - qmin)
    zero_point = qmin - (rmin / scale)
    zero_point = np.clip(round(zero_point), qmin, qmax).astype(np.int32)

    quantize_params = QuantizeAffineParams2(scale, zero_point, num_bits)
    # We must ensure that zero is exactly representable with these quantization
    # parameters. This is easy enough to add a self-check for.
    quantized_zero = quantize(np.array([0.0]), quantize_params)
    dequantized_zero = dequantize(quantized_zero, quantize_params)
    if dequantized_zero.item() != 0.0:
        raise ValueError(
            f"Quantization parameters are invalid: scale={scale}, zero={zero_point}. "
            f"Can't exactly represent zero."
        )

    return quantize_params


def quantize_affine_given_quant_params(
    input: torch.Tensor,
    quantize_params: QuantizeAffineParams2,
) -> torch.Tensor:
    """Get a quantizable approximation of a float tensor given quantize param.

    This function does not quantize the float tensor @input, but only adjusts it
    such that the returned float tensor has an exact quantized representation.
    This is a function that we want to use at training time to quantize biases
    and other parameters whose quantization schema is enforced by other
    parameteres.

    In forward pass, this function is equivalent to

        dequantize(get_quantized_representation(input, quantize_param))

    However, in backward pass, this function operates as identity, making it
    ideal to be a part of the training forward pass.
    """
    return QuantizeAffineFunction.apply(input, quantize_params)


def quantize(
    arr: np.ndarray, quantize_params: QuantizeAffineParams2
) -> np.ndarray:
    """Quantize a floating point array with respect to the quantization params.

    Arguments:
        arr (np.ndarray): The floating point data to quantize.
        quantize_params (QuantizeAffineParams): The quantization parameters
            under which the data should be quantized.
    """
    scale = quantize_params.scale
    zero_point = quantize_params.zero_point
    num_bits = quantize_params.num_bits
    if scale == 0:
        # Special case, all elements are zeros.
        if zero_point != 0:
            raise ValueError(
                "The given QuantizeAffineParams (={}) has a non-zero zero point"
                " with a scale of 0.".format(quantize_params)
            )
        return np.zeros_like(arr, dtype=np.int32)

    qmin, qmax = get_qmin_qmax(num_bits)
    reciprocal = 1 / scale
    quantized_values = (arr * reciprocal).round() + zero_point
    quantized_values = quantized_values.clip(qmin, qmax)
    return quantized_values


def dequantize(
    q_arr: np.ndarray,
    quantize_params: QuantizeAffineParams2,
) -> np.ndarray:
    """Dequantize a fixed point array with respect to the quantization params.

    Arguments:
        q_arr (np.ndarray): The quantized array to dequantize. It's dtype must
            match quantize_params.
        quantize_params (QuantizeAffineParams): The quantization parameters
            under which the data should be dequantized.
    """
    zero_point = quantize_params.zero_point
    scale = quantize_params.scale
    return (q_arr - zero_point) * scale


def quantize_affine(
    input: torch.Tensor,
    min_value: Optional[numbers.Real] = None,
    max_value: Optional[numbers.Real] = None,
    num_bits: int = None,
) -> torch.Tensor:
    """Return a quantizable approximation of a float tensor @input.

    This function does not quantize the float tensor @input, but only adjusts it
    such that the returned float tensor has an exact quantized representation.
    This is a function that we want to use at training time to quantize weights
    and activations.

    Arguments:
        input (Tensor): The input float Tensor to be quantized.
        min_value (scalar): The running min value (possibly averaged).
        max_value (scalar): The running max value (possibly averaged).
        num_bits (numpy.dtype): The number of bits.
    """
    if num_bits is None:
        raise ValueError("num_bits must be supplied")

    if min_value is None:
        # Force include 0 in our calculation of min_value.
        min_value = min(input.min().item(), 0.0)
    if max_value is None:
        # Force include 0 in our calculation of max_value.
        max_value = max(input.max().item(), 0.0)

    quantize_params = get_quantization_params(min_value, max_value, num_bits)
    return QuantizeAffineFunction.apply(input, quantize_params)


class QuantizeAffine(nn.Module):
    """Pytorch quantize_affine layer for quantizing layer outputs.

    This layer will keep a running max and min, which is used to compute a scale
    and zero_point for the quantization. Note that it is not always desirable
    to start the quantization immediately while training.

    Arguments:
        momentum (scalar): The amount of averaging of min and max bounds.
            This value should be in the range [0.0, 1.0].
        iteration_delay (scalar): The number of batches to wait before starting
            to quantize.
    """

    def __init__(
        self,
        momentum=0.1,
        iteration_delay=0,
        num_bits=8,
        quantizer_freeze_min_max=False,
    ):
        super().__init__()
        self.momentum = momentum
        self.iteration_delay = iteration_delay
        self.increment_counter = False
        self.num_bits = num_bits
        self.register_buffer("running_min_value", torch.tensor(0.0))
        self.register_buffer("running_max_value", torch.tensor(0.0))
        self.register_buffer(
            "iteration_count", torch.zeros([1], dtype=torch.int32).squeeze()
        )
        self.quantizer_freeze_min_max = quantizer_freeze_min_max

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(running_min="
            f"{self.running_min_value}, running_max="
            f"{self.running_max_value}, freeze_min_max="
            f"{self.quantizer_freeze_min_max}, num_bits={self.num_bits})"
        )

    def update_num_bits(self, num_bits):
        self.num_bits = num_bits

    def forward(self, input, recomp_bn_stats=False, override_alpha=False):
        if (
            self.training
            and self.is_active()
            and not self.quantizer_freeze_min_max
        ):
            # Force include 0 in min_value and max_value calculation.
            min_value = min(input.min().item(), 0)
            max_value = max(input.max().item(), 0)

            if self.iteration_count == self.iteration_delay:
                new_running_min_value = min_value
                new_running_max_value = max_value
            else:
                new_running_min_value = (
                    1.0 - self.momentum
                ) * self.running_min_value.item() + self.momentum * min_value
                new_running_max_value = (
                    1.0 - self.momentum
                ) * self.running_max_value.item() + self.momentum * max_value

            self.running_min_value.fill_(new_running_min_value)
            self.running_max_value.fill_(new_running_max_value)

        if self.is_active():
            output = quantize_affine(
                input,
                self.running_min_value.item(),
                self.running_max_value.item(),
                self.num_bits,
            )
        else:
            output = input

        if self.training and self.increment_counter:
            self.iteration_count.fill_(self.iteration_count.item() + 1)

        return output

    def is_active(self):
        if self.training:
            return self.iteration_count >= self.iteration_delay
        # If evaluating, always run quantization:
        return True
