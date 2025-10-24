from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
import jax.numpy as jnp

# Not the best implementation memory-wise, but it works for now.


class NoEquivR3Inputs(BaseThreewayInvariants):
    """Separating invariances of R3xR3 wrt {e}."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3 wrt {e}.

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]  # (B, S, 1, 3)
        y_bc = y[:, :, None, :]  # (B, S, 1, 3)

        # Broadcast p_pos
        p_pos_bc = jnp.zeros_like(p_pos[:, None, :, :])  # (B, 1, N, 3)

        rel_dists_x_2_lat = x_bc - p_pos_bc  # (B, S, N, 3)
        rel_dists_y_2_lat = y_bc - p_pos_bc  # (B, S, N, 3)

        # Concatenate along the last axis -> (B, S, N, 6)
        separating_inv = jnp.concatenate(
            [rel_dists_x_2_lat, rel_dists_y_2_lat], axis=-1
        )
        return separating_inv


class SymNoEquivR3Inputs(NoEquivR3Inputs):
    """Separating invariances of R3xR3 wrt {e} for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3 wrt {e} for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3).
        Returns:
            invariants (tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create the reordering index
        order = jnp.array([3, 4, 5, 0, 1, 2])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class NoEquivR2Inputs(BaseThreewayInvariants):
    """Separating invariances of R2xR2 wrt {e}."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2 wrt {e}.

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]  # (B, S, 1, 2)
        y_bc = y[:, :, None, :]  # (B, S, 1, 2)

        # Broadcast p_pos
        p_pos_bc = jnp.zeros_like(p_pos[:, None, :, :])  # (B, 1, N, 2)

        rel_dists_x_2_lat = x_bc - p_pos_bc  # (B, S, N, 2)
        rel_dists_y_2_lat = y_bc - p_pos_bc  # (B, S, N, 2)

        # Concatenate along the last axis -> (B, S, N, 4)
        separating_inv = jnp.concatenate(
            [rel_dists_x_2_lat, rel_dists_y_2_lat], axis=-1
        )
        return separating_inv


class SymNoEquivR2Inputs(NoEquivR2Inputs):
    """Separating invariances of R2xR2 wrt {e} for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2 wrt {e} for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2).
        Returns:
            invariants (tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create the reordering index
        order = jnp.array([2, 3, 0, 1])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)
