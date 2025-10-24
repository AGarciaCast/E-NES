"""
Utility functions for JAX/Flax neural networks.

This module provides utility functions for creating neural network layers with
specific initialization schemes and numerical stability helpers.
"""

import jax.numpy as jnp
import flax.linen as nn


def torch_compatible_dense(in_features, out_features):
    """
    Create a Dense layer with PyTorch-compatible initialization.

    This function creates a Flax Dense layer using initialization schemes
    that match PyTorch's default behavior for nn.Linear layers.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.

    Returns:
        nn.Dense: A Flax Dense layer with PyTorch-compatible initialization.
    """
    bound = 1 / jnp.sqrt(in_features) if in_features > 0 else 0

    return nn.Dense(
        features=out_features,
        kernel_init=nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform"),
        bias_init=nn.initializers.uniform(scale=bound),
    )


def ones_dense(in_features, out_features):
    """
    Create a Dense layer initialized with ones for weights and zeros for biases.

    Args:
        in_features: Number of input features (not used but kept for API consistency).
        out_features: Number of output features.

    Returns:
        nn.Dense: A Flax Dense layer with ones initialization for weights.
    """
    return nn.Dense(
        features=out_features,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros,
    )


def stable_norm(x, axis=-1, keepdims=False, epsilon=1e-12):
    """
    Compute a numerically stable vector norm.

    Adds a small epsilon before taking the square root to avoid zero gradients
    and numerical instability when the norm approaches zero.

    Args:
        x: Input array.
        axis: Axis along which to compute the norm. Default is -1.
        keepdims: Whether to keep the reduced dimensions. Default is False.
        epsilon: Small constant for numerical stability. Default is 1e-12.

    Returns:
        Array containing the computed norms with numerical stability.
    """
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + epsilon)


def safe_reciprocal(x, epsilon=1e-12):
    """
    Compute the reciprocal of x safely, avoiding division by zero.

    Args:
        x: Input array.
        epsilon: Small constant added to the denominator. Default is 1e-12.

    Returns:
        Array containing 1/(|x| + epsilon), avoiding division by zero.
    """
    return 1.0 / (jnp.abs(x) + epsilon)


def safe_div(x, y, epsilon=1e-12):
    """
    Perform safe division x/y avoiding division by zero.

    The function preserves the sign of y while adding epsilon to its absolute value.

    Args:
        x: Numerator array.
        y: Denominator array.
        epsilon: Small constant added to avoid division by zero. Default is 1e-12.

    Returns:
        Array containing x/y computed in a numerically stable way.
    """
    return x / (jnp.abs(y) + epsilon) * jnp.sign(y)


def stable_softmax(x, axis=-1):
    """
    Compute numerically stable softmax function.

    Subtracts the maximum value before computing exponentials to avoid
    numerical overflow issues with large input values.

    Args:
        x: Input array.
        axis: Axis along which to compute softmax. Default is -1.

    Returns:
        Array with softmax applied, normalized along the specified axis.
    """
    max_x = jnp.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / sum_exp


def safe_power(x, p, epsilon=1e-12):
    """
    Compute x^p safely, avoiding numerical issues with small numbers.

    Preserves the sign of x while adding epsilon to its absolute value
    before raising to the power p.

    Args:
        x: Input array (base).
        p: Power exponent.
        epsilon: Small constant added to avoid instability. Default is 1e-12.

    Returns:
        Array containing sign(x) * (|x| + epsilon)^p.
    """
    sign = jnp.sign(x)
    abs_x = jnp.abs(x) + epsilon
    result = sign * (abs_x**p)
    return result
