from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
import jax.numpy as jnp

##############################################################################
### S1 ###
##############################################################################


class OrthogonalSnInputsSnLatent(BaseThreewayInvariants):
    """Separating invariances of SnxSnxSn wrt O(n). For n<4"""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 3

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of SnxSnxSn wrt O(n).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, n).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, n).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 3).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        if x.shape[-1] == 1:
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
        elif x.shape[-1] == 2:
            # spherical coordinates to cartesian coordinates
            x_theta = x[..., 0]
            x_phi = x[..., 1]
            y_theta = y[..., 0]
            y_phi = y[..., 1]

            x_cos_t = jnp.cos(x_theta)
            x_sin_t = jnp.sin(x_theta)
            x_cos_p = jnp.cos(x_phi)
            x_sin_p = jnp.sin(x_phi)

            y_cos_t = jnp.cos(y_theta)
            y_sin_t = jnp.sin(y_theta)
            y_cos_p = jnp.cos(y_phi)
            y_sin_p = jnp.sin(y_phi)

            x = jnp.stack(
                [
                    x_cos_t * x_sin_p,
                    x_sin_t * x_sin_p,
                    x_cos_p,
                ],
                axis=-1,
            )
            y = jnp.stack(
                [
                    y_cos_t * y_sin_p,
                    y_sin_t * y_sin_p,
                    y_cos_p,
                ],
                axis=-1,
            )

        else:
            raise ValueError(
                f"Unsupported input dimension {x.shape[-1]}. Expected 1 or 2 for spherical coordinates."
            )

        # Broadcast x and y
        x_bc = x[:, :, None, :]  # (B, S, 1, n)
        y_bc = y[:, :, None, :]  # (B, S, 1, n)

        # Broadcast p_pos
        p_pos_bc = p_pos[:, None, :, :]  # (B, 1, N, n)

        sq_dists_x_2_lat = jnp.sum((x_bc - p_pos_bc) ** 2, axis=-1)  # (B, S, N)
        sq_dists_y_2_lat = jnp.sum((y_bc - p_pos_bc) ** 2, axis=-1)  # (B, S, N)

        # Compute squared distance between x and y for each pair and broadcast
        sq_dists_x_2_y = jnp.sum((x - y) ** 2, axis=-1)  # (B, S)
        sq_dists_x_2_y = jnp.expand_dims(sq_dists_x_2_y, axis=2)  # (B, S, 1)
        sq_dists_x_2_y = jnp.broadcast_to(
            sq_dists_x_2_y, sq_dists_x_2_lat.shape
        )  # (B, S, N)

        separating_inv = jnp.stack(
            [sq_dists_x_2_lat, sq_dists_y_2_lat, sq_dists_x_2_y], axis=-1
        )
        return separating_inv


class SymOrthogonalSnInputsSnLatent(OrthogonalSnInputsSnLatent):
    """Separating invariances of SnxSnxSn wrt O(n) for both inputs (x,y) and (y,x). For n<4"""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of SnxSnxSn wrt O(n) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, n).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, n).
        Returns:
            invariants (tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 3).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create the reordering index
        order = jnp.array([1, 0, 2])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialOrthogonalSnInputsSnLatent(BaseThreewayInvariants):
    """Separating invariances of SnxSnxSn wrt SO(n). For n<4"""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of SnxSnxSn wrt SO(n).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, n).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, n).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        if x.shape[-1] == 1:
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
        elif x.shape[-1] == 2:
            # spherical coordinates to cartesian coordinates
            x_theta = x[..., 0]
            x_phi = x[..., 1]
            y_theta = y[..., 0]
            y_phi = y[..., 1]

            x_cos_t = jnp.cos(x_theta)
            x_sin_t = jnp.sin(x_theta)
            x_cos_p = jnp.cos(x_phi)
            x_sin_p = jnp.sin(x_phi)

            y_cos_t = jnp.cos(y_theta)
            y_sin_t = jnp.sin(y_theta)
            y_cos_p = jnp.cos(y_phi)
            y_sin_p = jnp.sin(y_phi)

            x = jnp.stack(
                [
                    x_cos_t * x_sin_p,
                    x_sin_t * x_sin_p,
                    x_cos_p,
                ],
                axis=-1,
            )
            y = jnp.stack(
                [
                    y_cos_t * y_sin_p,
                    y_sin_t * y_sin_p,
                    y_cos_p,
                ],
                axis=-1,
            )

        else:
            raise ValueError(
                f"Unsupported input dimension {x.shape[-1]}. Expected 1 or 2 for spherical coordinates."
            )

        # Add broadcasting dimensions
        x_bc = x[:, :, None, :]  # (batch_size, num_sample_pairs, 1, n)
        y_bc = y[:, :, None, :]  # (batch_size, num_sample_pairs, 1, n)
        p_pos_bc = p_pos[:, None, :, :]  # (batch_size, 1, num_latents, n)

        x_lat_diff = x_bc - p_pos_bc
        y_lat_diff = y_bc - p_pos_bc

        sq_dists_x_2_lat = jnp.sum(x_lat_diff**2, axis=-1)
        sq_dists_y_2_lat = jnp.sum(y_lat_diff**2, axis=-1)

        # Calculate the squared distance between x and y points
        sq_dists_x_2_y = jnp.sum((x - y) ** 2, axis=-1)

        # Broadcast to match the shape of the other computations
        sq_dists_x_2_y = jnp.expand_dims(sq_dists_x_2_y, axis=2)
        sq_dists_x_2_y = jnp.broadcast_to(sq_dists_x_2_y, sq_dists_x_2_lat.shape)

        # Reshape to prepare for batched determinant calculation
        stacked_vectors = jnp.stack(
            [
                jnp.broadcast_to(x_bc, x_lat_diff.shape),
                jnp.broadcast_to(y_bc, x_lat_diff.shape),
                jnp.broadcast_to(p_pos_bc, x_lat_diff.shape),
            ],
            axis=-1,
        )

        # Compute determinant
        signed_simplex_volume = jnp.linalg.det(stacked_vectors)

        separating_inv = jnp.stack(
            [sq_dists_x_2_lat, sq_dists_y_2_lat, sq_dists_x_2_y, signed_simplex_volume],
            axis=-1,
        )

        return separating_inv


class SymSpecialOrthogonalSnInputsSnLatent(SpecialOrthogonalSnInputsSnLatent):
    """Separating invariances of SnxSnxSn wrt SO(n) for both inputs (x,y) and (y,x). For n<4"""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of SnxSnxSn wrt SO(n) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, n).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, n).
        Returns:
            invariants (tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create the reordering index
        order = jnp.array([1, 0, 2, 3])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        # Flip the sign of the determinant (last element)
        separating_inv_yx = separating_inv_yx.at[..., 3].multiply(-1.0)

        return (separating_inv_xy, separating_inv_yx)


class NoEquivS1InputsS1Latent(BaseThreewayInvariants):
    """Separating invariances of S1xS1 wrt {e}."""

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


class SymNoEquivS1InputsS1Latent(NoEquivS1InputsS1Latent):
    """Separating invariances of S1xS1 wrt {e} for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S1xS1 wrt {e} for (x,y) and (y,x).

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


class NoEquivS2InputsS2Latent(BaseThreewayInvariants):
    """Separating invariances of S1xS1 wrt {e}."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S2xS2 wrt {e}.

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

        x_theta = x[..., 0]
        x_phi = x[..., 1]
        y_theta = y[..., 0]
        y_phi = y[..., 1]

        x_cos_t = jnp.cos(x_theta)
        x_sin_t = jnp.sin(x_theta)
        x_cos_p = jnp.cos(x_phi)
        x_sin_p = jnp.sin(x_phi)

        y_cos_t = jnp.cos(y_theta)
        y_sin_t = jnp.sin(y_theta)
        y_cos_p = jnp.cos(y_phi)
        y_sin_p = jnp.sin(y_phi)

        x = jnp.stack(
            [
                x_cos_t * x_sin_p,
                x_sin_t * x_sin_p,
                x_cos_p,
            ],
            axis=-1,
        )
        y = jnp.stack(
            [
                y_cos_t * y_sin_p,
                y_sin_t * y_sin_p,
                y_cos_p,
            ],
            axis=-1,
        )

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


class SymNoEquivS2InputsS2Latent(NoEquivS2InputsS2Latent):
    """Separating invariances of S2xS2 wrt {e} for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S2xS2 wrt {e} for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
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
