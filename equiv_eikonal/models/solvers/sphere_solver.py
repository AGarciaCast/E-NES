from typing import Union
import jax.numpy as jnp
from equiv_eikonal.models.solvers.generic_riemannian_solver import (
    GenericEmbeddedRiemannianEquivariantNeuralEikonalSolver,
)

from equiv_eikonal.utils import safe_reciprocal


def spherical_distance_stable(points1, points2):
    """
    Compute spherical distance between two batches of points on the unit sphere.

    Args:
        points1: Array of shape (b, 3) - first batch of unit vectors
        points2: Array of shape (b, 3) - second batch of unit vectors

    Returns:
        Array of shape (b,) - spherical distances (angles in radians)
    """
    # Method 1: Using atan2 (most stable)
    # For unit vectors p1, p2: distance = atan2(||p1 × p2||, p1 · p2)

    # Compute dot product (cosine of angle)
    dot_product = jnp.sum(points1 * points2, axis=1)

    # Compute cross product magnitude (sine of angle)
    cross_product = jnp.cross(points1, points2)
    cross_magnitude = jnp.linalg.norm(cross_product, axis=1)

    # Use atan2 for stable computation
    distance = jnp.arctan2(cross_magnitude, dot_product)

    return distance


def spherical_distance_arcsin(points1, points2):
    """
    Alternative stable implementation using arcsin.

    Args:
        points1: Array of shape (b, 3) - first batch of unit vectors
        points2: Array of shape (b, 3) - second batch of unit vectors

    Returns:
        Array of shape (b,) - spherical distances (angles in radians)
    """
    # For unit vectors: distance = 2 * arcsin(||p1 - p2|| / 2)
    diff = points1 - points2
    diff_magnitude = jnp.linalg.norm(diff, axis=1)

    # This is stable because arcsin is well-conditioned in [0, 1]
    distance = 2 * jnp.arcsin(jnp.clip(diff_magnitude / 2, 0, 1))

    return distance


def spherical_distance_arccos_safe(points1, points2):
    """
    Safe arccos implementation with proper clipping.

    Args:
        points1: Array of shape (b, 3) - first batch of unit vectors
        points2: Array of shape (b, 3) - second batch of unit vectors

    Returns:
        Array of shape (b,) - spherical distances (angles in radians)
    """
    dot_product = jnp.sum(points1 * points2, axis=1)

    # Clip to valid domain with a small epsilon to avoid numerical issues
    eps = 1e-6
    dot_product = jnp.clip(dot_product, -1 + eps, 1 - eps)

    distance = jnp.arccos(dot_product)

    return distance


# Recommended: Use the atan2 version for maximum stability
def spherical_distance(points1, points2):
    """
    Recommended stable spherical distance implementation.
    """
    return spherical_distance_stable(points1, points2)


class SphereEquivariantNeuralEikonalSolver(
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
    distance_type: str = "ambient"  # 'ambient' or 'geodesic'

    def setup(self):
        super().setup()

    def metric(self, inputs):
        """Metric tensor of manifold.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
        """

        phi = inputs[..., 1]

        sin_p = jnp.sin(phi)

        aux_ones = jnp.ones_like(phi)
        aux_zeros = jnp.zeros_like(phi)

        # Rotation matrices around x, y, z axes
        res = jnp.stack(
            [
                jnp.stack([sin_p**2, aux_zeros], axis=-1),
                jnp.stack([aux_zeros, aux_ones], axis=-1),
            ],
            axis=-2,
        )
        return res

    def distance(self, inputs):
        """Homogenous solution, i.e., distance between points.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).

        Returns:
            dist: Euclidean pairwise distances.
                Shape (batch_size, num_sample_pairs, 1).
        """

        if self.distance_type == "ambient":
            dist = super().distance(inputs)
        else:
            inputs = self.fn_transform(inputs)

            # Calculate difference with higher precision
            x = inputs[:, :, 0, :]
            y = inputs[:, :, 1, :]

            # Use stable norm
            # inner_prod = jnp.einsum("...i,...i->...", x, y)
            # dist = jnp.arccos(jnp.clip(inner_prod, -1.0, 1.0))[..., None]

            dist = spherical_distance(x.reshape(-1, 3), y.reshape(-1, 3)).reshape(
                inputs.shape[0], inputs.shape[1], 1
            )

        return dist

    def inverse_metric(self, inputs):
        """Inverse metric tensor.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
        """
        phi = inputs[..., 1]

        sin_p = jnp.sin(phi)

        aux_ones = jnp.ones_like(phi)
        aux_zeros = jnp.zeros_like(phi)

        # Rotation matrices around x, y, z axes
        res = jnp.stack(
            [
                jnp.stack([safe_reciprocal(sin_p**2), aux_zeros], axis=-1),
                jnp.stack([aux_zeros, aux_ones], axis=-1),
            ],
            axis=-2,
        )
        return res

    def fn_transform(self, inputs):
        """Transform the inputs to the manifold.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 2).

        Returns:
            transformed_inputs: Transformed inputs on the manifold.
                Shape (batch_size, num_sample_pairs, 2, 3).
        """

        theta = inputs[..., 0]
        phi = inputs[..., 1]

        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        cos_p = jnp.cos(phi)
        sin_p = jnp.sin(phi)

        transformed_inputs = jnp.stack(
            [
                cos_t * sin_p,
                sin_t * sin_p,
                cos_p,
            ],
            axis=-1,
        )

        return transformed_inputs

    def jac_fn_transform(self, inputs):
        """Jacobian of transform function evaluated at the input points.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 2).

        Returns:
            jacobian: Jacobian matrix of the transformation.
                Shape (batch_size, num_sample_pairs, 2, 3, 2).
        """
        # Extract theta and phi for each point separately
        # inputs shape: (batch_size, num_sample_pairs, 2, 2)
        # inputs[..., :, 0] = theta for both points, inputs[..., :, 1] = phi for both points

        theta = inputs[..., 0]  # shape: (batch_size, num_sample_pairs, 2)
        phi = inputs[..., 1]  # shape: (batch_size, num_sample_pairs, 2)

        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        cos_p = jnp.cos(phi)
        sin_p = jnp.sin(phi)
        aux_zeros = jnp.zeros_like(phi)

        # Build jacobian for each point
        # For each point, we need a 3x2 jacobian matrix:
        # [∂x/∂θ, ∂x/∂φ]
        # [∂y/∂θ, ∂y/∂φ]
        # [∂z/∂θ, ∂z/∂φ]

        # Partial derivatives:
        # x = cos(θ)sin(φ) -> ∂x/∂θ = -sin(θ)sin(φ), ∂x/∂φ = cos(θ)cos(φ)
        # y = sin(θ)sin(φ) -> ∂y/∂θ = cos(θ)sin(φ),  ∂y/∂φ = sin(θ)cos(φ)
        # z = cos(φ)       -> ∂z/∂θ = 0,              ∂z/∂φ = -sin(φ)

        jacobian = jnp.stack(
            [
                jnp.stack([-sin_t * sin_p, cos_t * cos_p], axis=-1),  # ∂x/∂θ, ∂x/∂φ
                jnp.stack([cos_t * sin_p, sin_t * cos_p], axis=-1),  # ∂y/∂θ, ∂y/∂φ
                jnp.stack([aux_zeros, -sin_p], axis=-1),  # ∂z/∂θ, ∂z/∂φ
            ],
            axis=-2,
        )

        # Final shape should be: (batch_size, num_sample_pairs, 2, 3, 2)
        return jacobian

    def ambient2chart(self, inputs):
        """Ambient space to chart space transformation, i.e., inverse of fn_transform.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
        """
        phi = jnp.arccos(inputs[..., 2])  # cos(phi) = z
        theta = jnp.arctan2(inputs[..., 1], inputs[..., 0])

        # Use jnp.where instead of boolean indexing
        theta = jnp.where(theta < 0, theta + 2 * jnp.pi, theta)

        return jnp.stack(
            [
                theta,
                phi,
            ],
            axis=-1,
        )
