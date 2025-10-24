from typing import Union
import jax.numpy as jnp
from equiv_eikonal.models.solvers.euclidean_solver import (
    EuclidenEquivariantNeuralEikonalSolver,
)

from equiv_eikonal.utils import safe_reciprocal


class GenericEmbeddedRiemannianEquivariantNeuralEikonalSolver(
    EuclidenEquivariantNeuralEikonalSolver
):
    """Base Equivariant Neural Eikonal Solver.

    Args:
        num_hidden: The number of hidden units.
        num_heads: The number of attention heads.
        latent_dim: The dimensionality of the latent code.
        invariant: The invariant to use for the attention operation.
        embedding_type: The type of embedding to use. 'rff' or 'siren'.
        embedding_freq_multiplier: The frequency multiplier for the embedding.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        factored: Whether to use factored attention.
        epsilon: Parameter for the Riemannian metric.
        xi: Parameter for the Riemannian metric.
        theta_range: Range for theta parameter.
    """

    num_hidden: int
    num_heads: int
    latent_dim: int
    invariant: object  # BaseThreewayInvariants
    embedding_type: str
    embedding_freq_multiplier: Union[float, float]
    vmin: float = 0.0
    vmax: float = 1.0
    factored: bool = True

    def setup(self):
        super().setup()

    def metric(self, inputs):
        """Metric tensor of manifold.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        """
        raise NotImplementedError("metric method must be implemented.")

    def inverse_metric(self, inputs):
        """Inverse metric tensor.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        """
        return jnp.linalg.inv(self.metric(inputs))

    def fn_transform(self, inputs):
        """Transform the inputs to the manifold.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        Returns:
            transformed_inputs: Transformed inputs on the manifold.
                Shape (batch_size, num_sample_pairs, 2, ambient_space_dim).
        """
        raise NotImplementedError("fn_transform method must be implemented.")

    def jac_fn_transform(self, inputs):
        """Jacobian of transform function evaluated at the input points.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        Returns:
            transformed_inputs: Transformed inputs on the manifold.
                Shape (batch_size, num_sample_pairs, 2, ambient_space_dim, dim_signal).
        """
        raise NotImplementedError("jac_fn_transform method must be implemented.")

    def ambient2chart(self, inputs):
        """Ambient space to chart space transformation, i.e., inverse of fn_transform.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, ambient_space_dim).
        """
        raise NotImplementedError("ambient2chart method must be implemented.")

    def distance(self, inputs):
        """Euclidean distance between pairs of points in the ambient space.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        Returns:
            dist: euclidean pairwise distances.
                Shape (batch_size, num_sample_pairs, 1).
        """
        return super().distance(self.fn_transform(inputs))

    def times_and_gradients(self, inputs, p, a):
        """Computes gradient of traveltimes w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).

        Returns:
            times: euclidean pairwise distances. Shape (batch_size, num_sample_pairs).
            gradients: euclidean gradients wrt inputs. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        """

        times, euc_gradients = super().times_and_gradients(
            inputs,
            p,
            a,
        )

        gradients = jnp.einsum(
            "bspij,bspj->bspi", self.inverse_metric(inputs), euc_gradients
        )

        return times, gradients

    def times_grad_vel(self, inputs, p, a, aux_vel=False):
        """Computes traveltimes, gradients, and velocity w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p: The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, dim_orientation).
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            aux_vel: If true then also return inverse of velocities.
        """

        times, gradients = self.times_and_gradients(inputs, p, a)

        epsilon = 1e-12
        norm_grad = jnp.sqrt(
            jnp.einsum(
                "bspij,bspi, bspj->bsp", self.metric(inputs), gradients, gradients
            )
            + epsilon
        )

        # Calculate velocity with safe reciprocal
        vel = safe_reciprocal(norm_grad, epsilon=epsilon)

        if aux_vel:

            return times, gradients, norm_grad, vel
        else:

            return times, gradients, vel
