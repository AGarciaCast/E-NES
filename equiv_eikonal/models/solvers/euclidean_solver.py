from typing import Union

import jax
import jax.numpy as jnp
import flax.linen as nn

# Import modules
from equiv_eikonal.steerable_attention.threeway_invariants import (
    BaseThreewayInvariants,
)
from equiv_eikonal.models.solvers.__base_solver import EquivariantNeuralEikonalSolver

from equiv_eikonal.utils import stable_norm, safe_reciprocal, safe_div


class EuclidenEquivariantNeuralEikonalSolver(EquivariantNeuralEikonalSolver):
    """Base Equivariant Neural Eikonal Solver.

    Args:
        num_hidden: The number of hidden units.
        num_heads: The number of attention heads.
        latent_dim: The dimensionality of the latent code.
        invariant: The invariant to use for the attention operation.
        embedding_type: The type of embedding to use. 'rff' or 'siren'.
        embedding_freq_multiplier: The frequency multiplier for the embedding.
        vmin: Minimum value for output normalization.
        vmax: Maximum value for output normalization.
        factored: Whether to use factored representation.
    """

    num_hidden: int
    num_heads: int
    latent_dim: int
    invariant: BaseThreewayInvariants
    embedding_type: str
    embedding_freq_multiplier: Union[float, float]
    vmin: float = 0.0
    vmax: float = 1.0
    factored: bool = True

    def setup(self):
        super().setup()

    def distance(self, inputs):
        """Homogenous solution, i.e., distance between points.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        Returns:
            dist: Euclidean pairwise distances.
                Shape (batch_size, num_sample_pairs, 1).
        """

        # Calculate difference with higher precision
        diff = inputs[:, :, 0, :] - inputs[:, :, 1, :]

        # Use stable norm
        dist = stable_norm(diff, axis=-1, keepdims=True)

        return dist

    def project(self, inputs):
        """Project to manifold

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        """
        return inputs

    def times_and_gradients(self, inputs, p, a):
        """Computes gradient of traveltimes w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            reuse_grad: Parameter for controlling gradient computation.

        Returns:
            times: Euclidean pairwise distances. Shape (batch_size, num_sample_pairs).
            gradients: Euclidean gradients wrt inputs. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            forward_time: (Optional) Time taken for computation.
        """

        # Define gradient function using JAX's autodiff

        # Define a precision-aware value and gradient function
        def value_fn(inputs):
            return self(inputs, p, a)

        # Compute the value and gradient with automatic differentiation
        times, vjp_fn = jax.vjp(value_fn, inputs)

        # Apply VJP with ones to get gradients
        ones = jnp.ones_like(times, dtype=jnp.float32)
        gradients = vjp_fn(ones)[0]

        # Squeeze times and convert back to original dtype
        times = jnp.squeeze(times, axis=-1)

        return times, gradients

    def times_grad_vel(self, inputs, p, a, aux_vel=False):
        """Computes traveltimes, gradients, and velocity w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            reuse_grad: Parameter for controlling gradient computation.
            aux_vel: Whether to return inverse of velocities.
        """
        # Get times and gradients with enhanced precision
        outputs = self.times_and_gradients(inputs, p, a)
        times, gradients = outputs

        # Use stable norm calculation with epsilon
        epsilon = 1e-12
        norm_grad = stable_norm(gradients, axis=-1, epsilon=epsilon)

        # Calculate velocity with safe reciprocal
        vel = safe_reciprocal(norm_grad, epsilon=epsilon)

        if aux_vel:
            return times, gradients, norm_grad, vel
        else:
            return times, gradients, vel


class EuclideanFunctaNeuralEikonalSolver(EuclidenEquivariantNeuralEikonalSolver):
    """Euclidean Functa Neural Eikonal Solver.

    Args:
        num_hidden: The number of hidden units.
        latent_dim: The dimensionality of the latent code.
        invariant: The invariant to use for the attention operation.
        embedding_freq_multiplier: The frequency multiplier for the embedding.
        vmin: Minimum value for output normalization.
        vmax: Maximum value for output normalization.
        factored: Whether to use factored representation.
    """

    num_hidden: int
    latent_dim: int
    invariant: BaseThreewayInvariants
    embedding_freq_multiplier: float
    vmin: float = 0.0
    vmax: float = 1.0
    factored: bool = True

    def setup(self):
        from equiv_eikonal.models.functa import LatentModulatedSiren

        self.backbone = LatentModulatedSiren(
            num_in=4,
            num_hidden=128,
            num_layers=5,
            num_out=1,
            modulation_hidden_sizes=[32, 64, 128],
            w0=2.0,
        )

        # In Flax, parameters are typically initialized in __call__ or setup
        self.alpha = self.param("alpha", nn.initializers.ones, (1,))
        self.vmin_32 = jnp.asarray(self.vmin, jnp.float32)
        self.vmax_32 = jnp.asarray(self.vmax, jnp.float32)
