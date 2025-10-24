from typing import Union

import jax
import jax.numpy as jnp
import flax.linen as nn

from equiv_eikonal.steerable_attention.threeway_invariants import BaseThreewayInvariants
from equiv_eikonal.models.enf import EquivariantNeuralField
from equiv_eikonal.utils import safe_div


def pair_tensors(xs, xr):
    batchsize, nr, dim = xr.shape
    _, ns, _ = xs.shape

    # Reshape for broadcasting
    xs_exp = jnp.expand_dims(xs, axis=2)  # (batchsize, ns, 1, dim)
    xr_exp = jnp.expand_dims(xr, axis=1)  # (batchsize, 1, nr, dim)

    # Generate pairings
    paired = jnp.concatenate(
        (
            jnp.broadcast_to(xs_exp, (batchsize, ns, nr, dim)),
            jnp.broadcast_to(xr_exp, (batchsize, ns, nr, dim)),
        ),
        axis=-1,
    )  # (batchsize, ns, nr, 2*dim)

    # Reshape to desired output
    output = paired.reshape(batchsize, ns * nr, 2, dim)

    return output


class EquivariantNeuralEikonalSolver(nn.Module):
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
        self.backbone = EquivariantNeuralField(
            num_hidden=self.num_hidden,
            num_heads=self.num_heads,
            latent_dim=self.latent_dim,
            num_out=1,
            invariant=self.invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
        )

        # In Flax, parameters are typically initialized in __call__ or setup
        self.alpha = self.param("alpha", nn.initializers.ones, (1,))
        self.vmin_32 = jnp.asarray(self.vmin, jnp.float32)
        self.vmax_32 = jnp.asarray(self.vmax, jnp.float32)

    def __call__(self, inputs, p, a):
        """Computes traveltimes.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points. Shape of first component
                (batch_size, num_latents, dim_signal). Shape of second component
                (batch_size, num_latents, dim_signal, dim_orientation).
            a: The latent features. Shape (batch_size, num_latents, num_hidden).

        Returns:
            output: Shape (batch_size, num_sample_pairs).
            forward_time: (Optional) Time taken for forward pass.
        """

        # Apply backbone with higher precision
        out = self.backbone(inputs, p, a)

        # Apply adaptive sigmoid with careful clipping
        out = jax.nn.sigmoid(out * self.alpha)

        # Scale output with stable range conversion
        out = safe_div(
            (self.vmax_32 - self.vmin_32) * out + self.vmin_32,
            self.vmax_32 * self.vmin_32,
        )

        # Apply factored distance if needed
        if self.factored:
            out = self.distance(inputs) * out

        # out = self.distance(inputs)

        return out

    def times(self, inputs, p, a):
        """Computes traveltimes.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            record_time: Whether to record computation time.

        Returns:
            times: Shape (batch_size, num_sample_pairs).
            forward_time: (Optional) Time taken for forward pass.
        """
        return jnp.squeeze(self(inputs, p, a), axis=-1)

    def distance(self, inputs):
        """Homogenous solution, i.e., distance between points.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        """
        raise NotImplementedError("distance method must be implemented.")

    def norm_gradient(self, inputs, gradient):
        """Computes norm of gradient.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            gradient: Gradient to normalize.
        """
        raise NotImplementedError("norm_gradient method must be implemented.")

    def project(self, inputs):
        """Project to manifold.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
        """
        raise NotImplementedError("project method must be implemented.")

    def gradients(self, inputs, p, a):
        """Computes gradient of traveltimes w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p: The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, dim_orientation).
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            reuse_grad: Not used in JAX implementation.
        """
        _times, grads = self.times_and_gradients(inputs, p, a)

        # In JAX, there's no need to explicitly free memory
        return grads

    def times_and_gradients(self, inputs, p, a, reuse_grad=False, record_time=False):
        """Computes traveltimes, and gradient w.r.t. 'xs' and 'xr' given the time output.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            reuse_grad: Parameter for controlling gradient computation.
            record_time: Whether to record computation time.
        """
        raise NotImplementedError("times_and_gradients method must be implemented.")

    def times_grad_vel(
        self, inputs, p, a, reuse_grad=False, aux_vel=False, record_time=False
    ):
        """Computes traveltimes, gradients, and velocities w.r.t. 'xs' and 'xr'.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
            reuse_grad: Parameter for controlling gradient computation.
            aux_vel: Whether to return inverse of velocities.
            record_time: Whether to record computation time.
        """
        raise NotImplementedError("times_grad_vel method must be implemented.")

    def velocities(self, inputs, p, a):
        """Predicted velocity at 'xs' and 'xr.

        Args:
            inputs: The pose of the input points. Shape (batch_size, num_sample_pairs, 2, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).
        """
        _times, _grads, vel = self.times_grad_vel(
            inputs,
            p,
            a,
            aux_vel=False,
        )

        return vel

    def multisource(
        self,
        xs,
        xr,
        p,
        a,
    ):
        """Computes first-arrival traveltimes from complex source 'xs' (e.g. line).

        Arguments:
            xs: Source coordinates. Shape (batch_size, Ns, dim_signal).
            xr: Receiver points. Shape (batch_size, nr, dim_signal).
            p: The pose of the latent points.
            a: The latent features. Shape (batch_size, num_latents, num_hidden).

        Returns:
            T: Traveltimes from multisource 'xs' to receivers 'xr'. Shape (batch_size, nr)
            forward_time: (Optional) Time taken for forward pass.
        """
        X = pair_tensors(xs, xr)
        T = self.times(X, p, a)

        def more_than_one(_):
            batchsize, nr, dim = xr.shape
            _, ns, _ = xs.shape
            T = T.reshape(batchsize, ns, nr)
            T = jnp.min(T, axis=1)
            return T

        def just_one(_):
            return T

        # Handle multiple sources
        T = jax.lax.cond(xs.shape[1] > 1, more_than_one, just_one, None)

        return T
