from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
import jax.numpy as jnp

##############################################################################
### R2 ###
##############################################################################


class EuclideanR2InputsR2Latent(BaseThreewayInvariants):
    """Separating invariances of R2xR2xR2 wrt E(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 3

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xR2 wrt E(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 3).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]  # (B, S, 1, 2)
        y_bc = y[:, :, None, :]  # (B, S, 1, 2)

        # Broadcast p_pos
        p_pos_bc = p_pos[:, None, :, :]  # (B, 1, N, 2)

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


class SymEuclideanR2InputsR2Latent(EuclideanR2InputsR2Latent):
    """Separating invariances of R2xR2xR2 wrt E(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xR2 wrt E(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2).
        Returns:
            invariants (tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 3).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create the reordering index
        order = jnp.array([1, 0, 2])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR2InputsR2Latent(BaseThreewayInvariants):
    """Separating invariances of R2xR2xR2 wrt SE(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xR2 wrt SE(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Add broadcasting dimensions
        x_bc = x[:, :, None, :]  # (batch_size, num_sample_pairs, 1, 2)
        y_bc = y[:, :, None, :]  # (batch_size, num_sample_pairs, 1, 2)
        p_pos_bc = p_pos[:, None, :, :]  # (batch_size, 1, num_latents, 2)

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
        stacked_vectors = jnp.stack([x_lat_diff, y_lat_diff], axis=-1)

        # Compute determinant
        signed_simplex_volume = jnp.linalg.det(stacked_vectors)

        separating_inv = jnp.stack(
            [sq_dists_x_2_lat, sq_dists_y_2_lat, sq_dists_x_2_y, signed_simplex_volume],
            axis=-1,
        )

        return separating_inv


class SymSpecialEuclideanR2InputsR2Latent(SpecialEuclideanR2InputsR2Latent):
    """Separating invariances of R2xR2xR2 wrt SE(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xR2 wrt SE(2) for (x,y) and (y,x).

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
        order = jnp.array([1, 0, 2, 3])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        # Flip the sign of the determinant (last element)
        separating_inv_yx = separating_inv_yx.at[..., 3].multiply(-1.0)

        return (separating_inv_xy, separating_inv_yx)


##############################################################################
### R3 ###
##############################################################################


class EuclideanR3InputsR3Latent(BaseThreewayInvariants):
    """Separating invariances of R3xR3xR3 wrt E(3)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xR3 wrt E(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Add broadcasting dimensions
        x_bc = x[:, :, None, :]  # (batch_size, num_sample_pairs, 1, 3)
        y_bc = y[:, :, None, :]  # (batch_size, num_sample_pairs, 1, 3)
        p_pos_bc = p_pos[:, None, :, :]  # (batch_size, 1, num_latents, 3)

        # Compute the center
        center = (p_pos_bc + x_bc + y_bc) / 3

        # Now we have R3xR3xR3xR3 and we will compute their separating invariants wrt E(3)
        sq_dists_x_2_center = jnp.sum((x_bc - center) ** 2, axis=-1)
        sq_dists_y_2_center = jnp.sum((y_bc - center) ** 2, axis=-1)
        sq_dists_lat_2_center = jnp.sum((p_pos_bc - center) ** 2, axis=-1)
        sq_dists_x_2_lat = jnp.sum((x_bc - p_pos_bc) ** 2, axis=-1)
        sq_dists_y_2_lat = jnp.sum((y_bc - p_pos_bc) ** 2, axis=-1)

        # Calculate the squared distance between x and y points
        sq_dists_x_2_y = jnp.sum((x - y) ** 2, axis=-1)

        # Broadcast to match the shape of the other computations
        sq_dists_x_2_y = jnp.expand_dims(sq_dists_x_2_y, axis=2)
        sq_dists_x_2_y = jnp.broadcast_to(sq_dists_x_2_y, sq_dists_x_2_lat.shape)

        separating_inv = jnp.stack(
            [
                sq_dists_x_2_center,
                sq_dists_y_2_center,
                sq_dists_lat_2_center,
                sq_dists_x_2_lat,
                sq_dists_y_2_lat,
                sq_dists_x_2_y,
            ],
            axis=-1,
        )

        return separating_inv


class SymEuclideanR3InputsR3Latent(EuclideanR3InputsR3Latent):
    """Separating invariances of R3xR3xR3 wrt E(3) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xR3 wrt E(3) for (x,y) and (y,x).

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
        order = jnp.array([1, 0, 2, 4, 3, 5])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR3InputsR3Latent(BaseThreewayInvariants):
    """Separating invariances of R3xR3xR3 wrt SE(3)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 7

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xR3 wrt SE(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 7).
        """
        p_pos = p[0]
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Add broadcasting dimensions
        x_bc = x[:, :, None, :]  # (batch_size, num_sample_pairs, 1, 3)
        y_bc = y[:, :, None, :]  # (batch_size, num_sample_pairs, 1, 3)
        p_pos_bc = p_pos[:, None, :, :]  # (batch_size, 1, num_latents, 3)

        # Compute the center
        center = (p_pos_bc + x_bc + y_bc) / 3

        # Calculate differences from center
        x_center_diff = x_bc - center
        y_center_diff = y_bc - center
        lat_center_diff = p_pos_bc - center

        # Now we have R3xR3xR3xR3 and we will compute their separating invariants wrt E(3)
        sq_dists_x_2_center = jnp.sum(x_center_diff**2, axis=-1)
        sq_dists_y_2_center = jnp.sum(y_center_diff**2, axis=-1)
        sq_dists_lat_2_center = jnp.sum(lat_center_diff**2, axis=-1)
        sq_dists_x_2_lat = jnp.sum((x_bc - p_pos_bc) ** 2, axis=-1)
        sq_dists_y_2_lat = jnp.sum((y_bc - p_pos_bc) ** 2, axis=-1)

        # Calculate the squared distance between x and y points
        sq_dists_x_2_y = jnp.sum((x - y) ** 2, axis=-1)

        # Broadcast to match the shape of the other computations
        sq_dists_x_2_y = jnp.expand_dims(sq_dists_x_2_y, axis=2)
        sq_dists_x_2_y = jnp.broadcast_to(sq_dists_x_2_y, sq_dists_x_2_lat.shape)

        # Stack the vectors for determinant calculation
        stacked_vectors = jnp.stack(
            [x_center_diff, y_center_diff, lat_center_diff], axis=-1
        )

        # Compute determinant
        signed_simplex_volume = jnp.linalg.det(stacked_vectors)

        separating_inv = jnp.stack(
            [
                sq_dists_x_2_center,
                sq_dists_y_2_center,
                sq_dists_lat_2_center,
                sq_dists_x_2_lat,
                sq_dists_y_2_lat,
                sq_dists_x_2_y,
                signed_simplex_volume,
            ],
            axis=-1,
        )

        return separating_inv


class SymSpecialEuclideanR3InputsR3Latent(SpecialEuclideanR3InputsR3Latent):
    """Separating invariances of R3xR3xR3 wrt SE(3) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xR3 wrt SE(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3).
        Returns:
            invariants (tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 7).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create the reordering index
        order = jnp.array([1, 0, 2, 4, 3, 5, 6])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        # Flip the sign of the determinant (last element)
        separating_inv_yx = separating_inv_yx.at[..., 6].multiply(-1.0)

        return (separating_inv_xy, separating_inv_yx)
