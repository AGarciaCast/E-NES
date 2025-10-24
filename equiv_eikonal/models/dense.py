"""
Dense neural network modules with adaptive activation functions.

This module provides flexible dense neural network architectures with support for
various activation functions, including adaptive activation functions that can
learn scaling parameters during training.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, List, Union

from equiv_eikonal.utils import torch_compatible_dense, ones_dense

# ----------------------------
# Consistent initialization definitions.
# ----------------------------


# Define activation functions with jax.jit for better optimization
ACTS = {
    "tanh": jnp.tanh,
    "atan": jnp.arctan,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "relu": jax.nn.relu,
    "exp": jnp.exp,
    "elu": jax.nn.elu,
    "gelu": jax.nn.gelu,
    "sin": jnp.sin,
    "sinc": jax.jit(lambda z: jnp.where(z == 0, jnp.ones_like(z), jnp.sin(z) / z)),
    "linear": lambda z: z,
    "abs_linear": jnp.abs,
    "gauss": jax.jit(lambda z: jnp.exp(-(z**2))),
    "swish": jax.jit(lambda z: z * jax.nn.sigmoid(z)),
    "laplace": jax.jit(lambda z: jnp.exp(-jnp.abs(z))),
    "gauslace": jax.jit(lambda z: jnp.exp(-(z**2)) + jnp.exp(-jnp.abs(z))),
}


class AdaptiveActivation(nn.Module):
    """
    Layer for adaptive activation functions with learnable scaling.

    This module applies an activation function with an optional learnable
    scaling parameter 'a' that can be trained to adjust the activation strength.

    Attributes:
        act_name: Name of the activation function.
        adapt: Whether to use a learnable scaling parameter.
        n: Fixed scaling factor applied to inputs.
        act: The activation function to apply.
    """

    act_name: str
    adapt: bool
    n: float
    act: Callable

    def setup(self):
        """Initialize the learnable parameter if adaptive mode is enabled."""
        # Initialize parameter in setup if adaptive
        if self.adapt:
            self.a = self.param("a", nn.initializers.ones, (1,))

    @nn.compact
    def __call__(self, x):
        """
        Apply the activation function to the input.

        Args:
            x: Input array.

        Returns:
            Activated output, optionally scaled by learnable parameter.
        """
        # Use functional pattern for conditional logic
        if self.adapt:
            return self.act(self.n * self.a * x)
        else:
            return self.act(self.n * x)


class ActivationFactory:
    """
    Factory for creating activation functions from string specifications.

    This factory supports both simple activations (e.g., 'tanh', 'relu') and
    adaptive activations with learnable parameters (e.g., 'ad-gauss-1').
    """

    @staticmethod
    def create_activation(act_spec):
        """
        Create activation function from specification string or callable.

        The string format for adaptive activations is: '(ad)-activation_name-n'
        where 'ad' indicates adaptive (learnable scaling), 'activation_name' is
        the base activation function, and 'n' is a scaling factor.

        Args:
            act_spec: Either a callable or a string specification of the activation.

        Returns:
            A callable activation function or a factory for adaptive activations.

        Raises:
            ValueError: If the activation specification is invalid.

        Examples:
            - 'relu': Standard ReLU activation
            - 'ad-gauss-1': Adaptive Gaussian activation with scale 1
            - 'tanh': Standard tanh activation
        """
        if callable(act_spec):
            return act_spec

        if not isinstance(act_spec, str):
            raise ValueError("'act' must be either a 'str' or a 'callable'")

        # Handle direct activation lookup
        if "-" not in act_spec:
            if act_spec in ACTS:
                return ACTS[act_spec]
            raise ValueError(f"Unsupported activation: {act_spec}")

        # Parse adaptive activation
        parts = act_spec.split("-")
        if len(parts) != 3:
            raise ValueError(
                "Adaptive activation format should be '(ad)-activation_name-n'"
            )

        adapt = parts[0] == "ad"
        act_name = parts[1]

        if act_name not in ACTS:
            raise ValueError(f"Unsupported activation: {act_name}")

        try:
            n = float(parts[2])
        except ValueError:
            n = 1.0

        act_func = ACTS[act_name]

        # Return a factory function that creates the module when called
        return lambda x: AdaptiveActivation(
            act_name=act_name, adapt=adapt, n=n, act=act_func
        )(x)


# Function for backward compatibility
def Activation(act):
    """
    Backward-compatible interface for activation creation.

    Args:
        act: Activation specification (string or callable).

    Returns:
        Activation function created by ActivationFactory.
    """
    return ActivationFactory.create_activation(act)


class DenseBody(nn.Module):
    """
    Multi-layer dense neural network with configurable architecture.

    This module creates a fully-connected neural network with customizable
    hidden layer dimensions, activation functions, and output dimensions.

    Attributes:
        input_dim: Dimensionality of the input features.
        nu: Number of hidden units. Can be an int (same for all layers) or a
            list of ints (one per layer).
        nl: Number of hidden layers.
        out_dim: Dimensionality of the output. Default is 1.
        act: Activation function for hidden layers. Default is 'ad-gauss-1'.
        out_act: Activation function for output layer. Default is 'linear'.
    """

    input_dim: int
    nu: Union[int, List[int]]
    nl: int
    out_dim: int = 1
    act: Union[str, Callable] = "ad-gauss-1"
    out_act: Union[str, Callable] = "linear"

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the dense network.

        Args:
            x: Input array of shape (..., input_dim).

        Returns:
            Output array of shape (..., out_dim).

        Raises:
            AssertionError: If nu is a list/tuple but its length doesn't match nl.
        """
        # Process nu parameter
        if isinstance(self.nu, int):
            hidden_dims = [self.nu] * self.nl
        else:
            assert (
                isinstance(self.nu, (list, tuple)) and len(self.nu) == self.nl
            ), "Number of hidden layers 'nl' must match the length of 'nu'"
            hidden_dims = self.nu

        # Define weight initializer (kaiming_normal / He initialization)
        kernel_init = nn.initializers.variance_scaling(2.0, "fan_in", "normal")
        bias_init = nn.initializers.zeros

        # Input layer
        x = torch_compatible_dense(
            in_features=self.input_dim, out_features=hidden_dims[0]
        )(x)
        x = Activation(self.act)(x)

        # Hidden layers
        for i in range(1, self.nl):
            x = nn.Dense(
                features=hidden_dims[i],
                kernel_init=kernel_init,
                bias_init=bias_init,
                name=f"hidden_layer_{i}",
            )(x)

            x = Activation(self.act)(x)

        # Output layer
        x = nn.Dense(
            features=self.out_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
            name="output_layer",
        )(x)
        x = Activation(self.out_act)(x)

        return x
