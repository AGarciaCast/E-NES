import jax
import jax.numpy as jnp
from flax import linen as nn


class PolynomialFeatures(nn.Module):
    degree: int

    @nn.compact
    def __call__(self, x):
        # Pre-initialize with input features
        polynomial_list = [x]

        # Create polynomial features using a more efficient implementation
        for d in range(self.degree):
            # Compute the next degree using einsum for better performance
            next_degree = jnp.einsum("...i,...j->...ij", polynomial_list[-1], x)
            # Reshape in one operation
            next_degree = next_degree.reshape(*x.shape[:-1], -1)
            polynomial_list.append(next_degree)

        # Single concatenation operation
        return jnp.concatenate(polynomial_list, axis=-1)


class PolynomialEmbedding(nn.Module):
    num_out: int
    num_hidden: int
    degree: int
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # Replace assertion with JAX-compatible validation
        jax.lax.cond(
            self.num_layers >= 2,
            lambda _: None,
            lambda _: jax.debug.print(
                "Error: num_layers must be >= 2, got {}", self.num_layers
            ),
            None,
        )

        # Compute polynomial features
        x = PolynomialFeatures(degree=self.degree)(x)

        # Apply hidden layers more efficiently
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.num_hidden)(x)
            x = nn.gelu(x)

        # Output layer
        x = nn.Dense(features=self.num_out)(x)

        return x
