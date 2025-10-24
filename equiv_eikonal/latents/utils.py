import math
import jax.numpy as jnp
import jax.random as random
import jax
from jax.nn import sigmoid  # Import sigmoid from jax.nn


def init_positions_grid(num_latents: int, num_signals: int, num_dims: int, xmin, xmax):
    """
    Initialize the latent poses on a grid using JAX.

    Args:
         num_latents (int): The number of latent points.
        dim_signal (int): The number of signals.
        num_pos_dims (int): The number of position dimensions.
        num_ori_dims (int): The number of orientation dimensions.

    Returns:
        z_positions (jax.numpy.ndarray): The latent poses for each signal. Shape [num_signals, num_latents, ...].
    """

    # Ensure num_latents is a power of num_dims
    assert (
        abs(round(num_latents ** (1.0 / num_dims), 5) % 1) < 1e-5
    ), "num_latents must be a power of the number of position dimensions"

    # Calculate the number of latents per dimension
    num_latents_per_dim = int(round(num_latents ** (1.0 / num_dims)))

    # Create an n-d mesh-grid [-1 to 1]
    # Generate linspace for each dimension
    linspace_per_dim = [
        jnp.linspace(
            xmin[i] + 1 / num_latents_per_dim,
            xmax[i] - 1 / num_latents_per_dim,
            num_latents_per_dim,
        )
        for i in range(num_dims)
    ]

    # Create an n-dimensional mesh grid
    positions = jnp.stack(
        jnp.meshgrid(*linspace_per_dim, indexing="ij"),  # Use 'ij' for matrix indexing
        axis=-1,
    ).reshape(-1, num_dims)

    # Repeat for number of signals
    positions = jnp.repeat(positions[None, :, :], num_signals, axis=0)

    return positions


def init_orientations_fixed(num_latents: int, num_signals: int, num_dims: int):
    """Initialize the latent orientations as fixed.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.

    Returns:
        z_orientations (torch.Tensor): The latent orientations for each signal. Shape [num_signals, num_latents, ...].
    """
    orientations = jnp.zeros((num_signals, num_latents, num_dims))
    return orientations


def init_orientations_random_uniform(
    key, num_latents, num_signals, num_dims, norm=False
):
    """Initialize the latent orientations randomly using JAX.

    Args:
        key (jax.random.PRNGKey): Random key for JAX's random number generator.
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        num_dims (int): The number of dimensions.
        norm (bool): Whether to normalize the orientations to [-1, 1].

    Returns:
        jnp.ndarray: The latent orientations. Shape [num_signals, num_latents, num_dims].
    """
    key, subkey = random.split(key)  # Split key for new randomness
    orientations = random.uniform(subkey, (num_signals, num_latents, num_dims))

    # Replace lax.cond with direct operations
    if norm:
        # Apply normalization to [-1, 1]
        orientations = orientations * 2.0 - 1.0
    else:
        # Handle the 2D case separately
        if num_dims == 2:
            # First dimension scaled by π
            orientations = orientations.at[..., 0].set(orientations[..., 0] * jnp.pi)
            # Second dimension scaled by 2π
            orientations = orientations.at[..., 1].set(
                orientations[..., 1] * 2 * jnp.pi
            )
        else:
            # For other dimensions, scale to [-π, π]
            orientations = orientations * 2 * jnp.pi - jnp.pi

    return orientations


def init_appearances_ones(num_latents: int, num_signals: int, latent_dim: int):
    """Initialize the latent features as ones.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        latent_dim (int): The dimensionality of the latent code.

    Returns:
        z_features (torch.Tensor): The latent features for each signal. Shape [num_signals, num_latents, latent_dim].
    """
    z_features = jnp.ones((num_signals, num_latents, latent_dim))
    return z_features


def init_appearances_mean(num_latents: int, num_signals: int, latent_dim: int):
    """Initialize the latent features as ones.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        latent_dim (int): The dimensionality of the latent code.

    Returns:
        z_features (torch.Tensor): The latent features for each signal. Shape [num_signals, num_latents, latent_dim].
    """
    z_features = jnp.ones((num_signals, num_latents, latent_dim)) / latent_dim
    return z_features


def soft_clip(x, min_val, max_val, alpha=10.0):
    """Soft clipping that preserves gradients.

    Args:
        x: Input tensor to clip
        min_val: Minimum value (soft boundary)
        max_val: Maximum value (soft boundary)
        alpha: Controls the sharpness of the transition (higher = closer to hard clipping)

    Returns:
        Tensor with values softly restricted to the [min_val, max_val] range.
    """
    # Transform to normalized space
    range_width = max_val - min_val
    normalized = (x - min_val) / range_width

    # Apply smooth sigmoid-based clipping
    clamped = sigmoid(alpha * (normalized - 0.5)) * 0.998 + 0.001

    # Transform back to original space
    return min_val + clamped * range_width
