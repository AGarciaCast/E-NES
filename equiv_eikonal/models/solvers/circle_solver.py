from typing import Union
import jax.numpy as jnp
from equiv_eikonal.models.solvers.generic_riemannian_solver import (
    GenericEmbeddedRiemannianEquivariantNeuralEikonalSolver,
)


class CircleEquivariantNeuralEikonalSolver(
    GenericEmbeddedRiemannianEquivariantNeuralEikonalSolver
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
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 1).
        """

        return jnp.expand_dims(jnp.ones_like(inputs), axis=-1)

    def inverse_metric(self, inputs):
        """Inverse metric tensor.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 1).
        """
        return jnp.expand_dims(jnp.ones_like(inputs), axis=-1)

    def fn_transform(self, inputs):
        """Transform the inputs to the manifold.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 1).

        Returns:
            transformed_inputs: Transformed inputs on the manifold.
                Shape (batch_size, num_sample_pairs, 2, 2).
        """
        x = inputs[:, :, 0, :]  # (B, S, n)
        y = inputs[:, :, 1, :]  # (B, S, n)
        x_theta = x.squeeze(-1)
        y_theta = y.squeeze(-1)
        x = jnp.stack(
            [
                jnp.cos(x_theta),
                jnp.sin(x_theta),
            ],
            axis=-1,
        )

        y = jnp.stack(
            [
                jnp.cos(y_theta),
                jnp.sin(y_theta),
            ],
            axis=-1,
        )
        transformed_inputs = jnp.stack([x, y], axis=2)
        return transformed_inputs
