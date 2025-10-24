from typing import Union
import jax.numpy as jnp
from equiv_eikonal.models.solvers.euclidean_solver import (
    EuclidenEquivariantNeuralEikonalSolver,
)

from equiv_eikonal.utils import safe_power, safe_reciprocal, safe_div

# Constant for sub-Riemannian approximation. Based on https://arxiv.org/abs/2210.00935
NU = 1.6


class PositionOrientationEquivariantNeuralEikonalSolver(
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
    epsilon: float = 1.0
    xi: float = 1.0
    theta_range: Union[float, float] = (0.0, 2.0 * jnp.pi)

    def setup(self):
        super().setup()

    def distance(self, inputs):
        """Homogenous solution, i.e., distance between points.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 3).

        Returns:
            dist: euclidean pairwise distances.
                Shape (batch_size, num_sample_pairs, 1).
        """
        g1, g2 = inputs[:, :, 0, :], inputs[:, :, 1, :]
        g0 = jnp.zeros_like(g2)

        g1_theta = g1[..., 2]
        g1_pos = g1[..., :2]
        g2_theta = g2[..., 2]
        g2_pos = g2[..., :2]

        # Rotation matrix for -g1_theta
        cos_t = jnp.cos(g1_theta)  # cos(-θ) = cos(θ)
        sin_t = jnp.sin(g1_theta)  # sin(-θ) = -sin(θ)

        # Create rotation matrices for -g1_theta
        R = jnp.stack(
            [
                jnp.stack([cos_t, -sin_t], axis=-1),  # Note: sin has negative sign
                jnp.stack([sin_t, cos_t], axis=-1),
            ],
            axis=-2,
        )

        # Computing invariants - note the order is reversed
        pos_dif = g2_pos - g1_pos

        # Use jax.lax.batch_matmul for the einsum operation
        g0 = g0.at[..., :2].set(jnp.einsum("bsij,bsj->bsi", R, pos_dif))
        diff_theta = g2_theta - g1_theta  # Note the order is reversed
        g0 = g0.at[..., 2].set(
            jnp.remainder(diff_theta + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        )

        g0_x = g0[..., 0]
        g0_y = g0[..., 1]
        g0_theta = g0[..., 2]

        w1 = self.xi
        w3 = 1

        euc_dist = jnp.expand_dims(
            jnp.sqrt((w1 * g0_x) ** 2 + (w1 * g0_y) ** 2 + (w3 * g0_theta) ** 2 + 1e-8),
            axis=2,
        )

        b1 = g0_x * jnp.cos(g0_theta * 0.5) + g0_y * jnp.sin(g0_theta * 0.5)
        b2 = -g0_x * jnp.sin(g0_theta * 0.5) + g0_y * jnp.cos(g0_theta * 0.5)
        b3 = g0_theta

        sub_riem_dist = jnp.expand_dims(
            safe_power(
                (NU * (w1 + w3)) ** 4 * b2**2 + ((w1 * b1) ** 2 + (w3 * b3) ** 2) ** 2,
                1 / 4,
                1e-12,
            ),
            axis=2,
        )

        if self.epsilon > 0:
            w2 = safe_div(self.xi, self.epsilon, 1e-12)
            riem_dist = jnp.expand_dims(
                jnp.sqrt((w1 * b1) ** 2 + (w2 * b2) ** 2 + (w3 * b3) ** 2 + 1e-12),
                axis=2,
            )

            local_dist = jnp.concatenate(
                [
                    sub_riem_dist,
                    riem_dist,
                ],
                axis=-1,
            )

            local_dist = jnp.min(local_dist, axis=-1, keepdims=True)

        else:
            local_dist = sub_riem_dist

        global_local_dist = jnp.concatenate(
            [
                euc_dist,
                local_dist,
            ],
            axis=-1,
        )

        dist = jnp.max(global_local_dist, axis=-1, keepdims=True)

        return dist

    def project(self, inputs):
        """Project to manifold

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
        """

        # Update the angle component to be within the specified range
        res = inputs.at[..., 2].set(
            jnp.remainder(inputs[..., 2] - self.theta_range[0], 2.0 * jnp.pi)
            + self.theta_range[0]
        )
        return res

    def times_and_gradients(self, inputs, p, a):
        """Computes gradient of traveltimes w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p: The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, dim_orientation).
            a: The latent features. Shape (batch_size, num_latents, num_hidden).

        Returns:
            times: euclidean pairwise distances. Shape (batch_size, num_sample_pairs).
            gradients: euclidean gradients wrt inputs. Shape (batch_size, num_sample_pairs, 2, 3).
        """

        times, euc_gradients = super().times_and_gradients(
            inputs,
            p,
            a,
        )

        theta = inputs[..., 2]

        # Compute sine and cosine values
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        aux_zeros = jnp.zeros_like(theta)
        aux_ones = jnp.ones_like(theta)

        rot_t = jnp.stack(
            [
                jnp.stack([cos_t, sin_t, aux_zeros], axis=-1),
                jnp.stack([-sin_t, cos_t, aux_zeros], axis=-1),
                jnp.stack([aux_zeros, aux_zeros, aux_ones], axis=-1),
            ],
            axis=-2,
        )

        oriented_gradients = jnp.einsum("bspij,bspj->bspi", rot_t, euc_gradients)

        inverse_metric = jnp.diag(
            jnp.array(
                [1.0 / (self.xi**2), (self.epsilon**2) / (self.xi**2), 1.0],
                dtype=jnp.float32,
            )
        )

        gradients_coords = jnp.einsum(
            "ij,bspj->bspi", inverse_metric, oriented_gradients
        )

        rot = jnp.transpose(rot_t, axes=(0, 1, 2, 4, 3))

        gradients = jnp.einsum("bspij,bspj->bspi", rot, gradients_coords)

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

        times, euc_gradients = super().times_and_gradients(inputs, p, a)

        theta = inputs[..., 2]

        # Compute sine and cosine values
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        aux_zeros = jnp.zeros_like(theta)
        aux_ones = jnp.ones_like(theta)

        rot_t = jnp.stack(
            [
                jnp.stack([cos_t, sin_t, aux_zeros], axis=-1),
                jnp.stack([-sin_t, cos_t, aux_zeros], axis=-1),
                jnp.stack([aux_zeros, aux_zeros, aux_ones], axis=-1),
            ],
            axis=-2,
        )

        oriented_gradients = jnp.einsum("bspij,bspj->bspi", rot_t, euc_gradients)

        inverse_metric = jnp.diag(
            jnp.array(
                [1.0 / (self.xi**2), (self.epsilon**2) / (self.xi**2), 1.0],
                dtype=jnp.float32,
            )
        )

        gradients_coords = jnp.einsum(
            "ij,bspj->bspi", inverse_metric, oriented_gradients
        )

        rot = jnp.transpose(rot_t, axes=(0, 1, 2, 4, 3))

        gradients = jnp.einsum("bspij,bspj->bspi", rot, gradients_coords)

        epsilon = 1e-12
        norm_grad = jnp.sqrt(
            jnp.einsum("bspi,bspi->bsp", euc_gradients, gradients) + epsilon
        )

        # Calculate velocity with safe reciprocal
        vel = safe_reciprocal(norm_grad, epsilon=epsilon)

        if aux_vel:

            return times, gradients, norm_grad, vel
        else:

            return times, gradients, vel
