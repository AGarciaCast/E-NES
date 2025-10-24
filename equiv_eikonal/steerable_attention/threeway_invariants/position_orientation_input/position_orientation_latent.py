from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
import jax
import jax.numpy as jnp


##############################################################################
### R2 ###
##############################################################################


class EuclideanR2xS1InputsR2xS1Latent(BaseThreewayInvariants):
    """Separating invariances of R2xR2x(R2xS1) wrt E(2)."""

    def setup(self):
        super().setup()
        """Setup method for Flax modules."""
        self.dim = 14

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2x(R2xS1) wrt E(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 14).
        """
        p_pos, p_ori = p
        x_pos = inputs[:, :, 0, :2]
        y_pos = inputs[:, :, 1, :2]
        x_theta = inputs[:, :, 0, 2]
        y_theta = inputs[:, :, 1, 2]

        x_ori = jnp.stack(
            [
                jnp.cos(x_theta),
                jnp.sin(x_theta),
            ],
            axis=-1,
        )

        y_ori = jnp.stack(
            [
                jnp.cos(y_theta),
                jnp.sin(y_theta),
            ],
            axis=-1,
        )

        # Broadcast x and y
        x_pos_bc = jnp.expand_dims(x_pos, axis=2)  # (batch, samples, 1, 2)
        y_pos_bc = jnp.expand_dims(y_pos, axis=2)  # (batch, samples, 1, 2)
        x_ori_bc = jnp.expand_dims(x_ori, axis=2)  # (batch, samples, 1, 2)
        y_ori_bc = jnp.expand_dims(y_ori, axis=2)  # (batch, samples, 1, 2)

        # Broadcast p_pos
        p_pos_bc = jnp.expand_dims(p_pos, axis=1)  # (batch, 1, latents, 2)

        # Reduction to the isotropy
        x_lat_diff = x_pos_bc - p_pos_bc
        y_lat_diff = y_pos_bc - p_pos_bc

        # Create transformation matrix A
        perpendicular_ori = jnp.stack([-1.0 * p_ori[..., 1], p_ori[..., 0]], axis=-1)
        A = jnp.stack([p_ori, perpendicular_ori], axis=-1)
        A = jnp.transpose(A, (0, 1, 3, 2))  # Transpose last two dimensions

        # Apply transformations using jax.lax.batch_matmul or einsum
        iso_x_pos = jnp.einsum("blij,bslj->bsli", A, x_lat_diff)
        iso_y_pos = jnp.einsum("blij,bslj->bsli", A, y_lat_diff)
        iso_x_ori = jnp.einsum("blij,bslj->bsli", A, x_ori_bc)
        iso_y_ori = jnp.einsum("blij,bslj->bsli", A, y_ori_bc)

        # Compute the invariants of (R2xS1)x(R2xS1) wrt O(1)
        invariant1 = iso_x_pos[..., 0]
        invariant2 = (iso_x_pos[..., 1]) ** 2
        invariant3 = iso_x_ori[..., 0]
        invariant4 = (iso_x_ori[..., 1]) ** 2
        invariant5 = (iso_x_pos[..., 1] - iso_x_ori[..., 1]) ** 2
        invariant6 = iso_y_pos[..., 0]
        invariant7 = (iso_y_pos[..., 1]) ** 2
        invariant8 = iso_y_ori[..., 0]
        invariant9 = (iso_y_ori[..., 1]) ** 2
        invariant10 = (iso_y_pos[..., 1] - iso_y_ori[..., 1]) ** 2
        invariant11 = (iso_x_pos[..., 1] - iso_y_pos[..., 1]) ** 2
        invariant12 = (iso_x_pos[..., 1] - iso_y_ori[..., 1]) ** 2
        invariant13 = (iso_x_ori[..., 1] - iso_y_pos[..., 1]) ** 2
        invariant14 = (iso_x_ori[..., 1] - iso_y_ori[..., 1]) ** 2

        separating_inv = jnp.stack(
            [
                invariant1,
                invariant2,
                invariant3,
                invariant4,
                invariant5,
                invariant6,
                invariant7,
                invariant8,
                invariant9,
                invariant10,
                invariant11,
                invariant12,
                invariant13,
                invariant14,
            ],
            axis=-1,
        )

        return separating_inv


class SymEuclideanR2xS1InputsR2xS1Latent(EuclideanR2xS1InputsR2xS1Latent):
    """Separating invariances of R2xR2x(R2xS1) wrt E(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2x(R2xS1) wrt E(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 28).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create reordering indices
        order = jnp.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 10, 11, 12, 13])

        # Use take to reorder along the last dimension
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR2xS1InputsR2xS1Latent(BaseThreewayInvariants):
    """Separating invariances of (R2xS1)x(R2xS1)x(R2xS1) wrt SE(2)."""

    def setup(self):
        super().setup()
        """Setup method for Flax modules."""
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of (R2xS1)x(R2xS1)x(R2xS1) wrt SE(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        p_pos, p_ori = p
        x_pos = inputs[:, :, 0, :2]
        y_pos = inputs[:, :, 1, :2]
        x_theta = inputs[:, :, 0, 2]
        y_theta = inputs[:, :, 1, 2]

        # Fixed variable reference: p_ori_theta wasn't defined before using iso_x_ori
        # Instead, compute it directly from p_ori
        p_ori_theta = jnp.arctan2(p_ori[..., 1], p_ori[..., 0])

        # Broadcast x and y
        x_pos_bc = jnp.expand_dims(x_pos, axis=2)  # (batch, samples, 1, 2)
        y_pos_bc = jnp.expand_dims(y_pos, axis=2)  # (batch, samples, 1, 2)

        # Broadcast p_pos
        p_pos_bc = jnp.expand_dims(p_pos, axis=1)  # (batch, 1, latents, 2)

        x_lat_diff = x_pos_bc - p_pos_bc
        y_lat_diff = y_pos_bc - p_pos_bc

        # Create transformation matrix A
        perpendicular_ori = jnp.stack([-1.0 * p_ori[..., 1], p_ori[..., 0]], axis=-1)
        A = jnp.stack([p_ori, perpendicular_ori], axis=-1)
        A = jnp.transpose(A, (0, 1, 3, 2))  # Transpose last two dimensions

        # Apply transformations
        iso_x_pos = jnp.einsum("blij,bslj->bsli", A, x_lat_diff)
        iso_y_pos = jnp.einsum("blij,bslj->bsli", A, y_lat_diff)

        # Handle angular differences with proper modulo
        p_ori_theta_bc = jnp.expand_dims(
            p_ori_theta, axis=(1, 3)
        )  # (batch, 1, latents, 1)

        # Calculate angular differences and normalize
        x_theta_bc = jnp.expand_dims(x_theta, axis=(2, 3))  # (batch, samples, 1, 1)
        y_theta_bc = jnp.expand_dims(y_theta, axis=(2, 3))  # (batch, samples, 1, 1)

        iso_x_ori = (
            jnp.remainder((x_theta_bc - p_ori_theta_bc) + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        )

        iso_y_ori = (
            jnp.remainder((y_theta_bc - p_ori_theta_bc) + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        )

        # Concatenate all invariants
        separating_inv = jnp.concatenate(
            [
                iso_x_pos,
                iso_x_ori,
                iso_y_pos,
                iso_y_ori,
            ],
            axis=-1,
        )

        return separating_inv


class SymSpecialEuclideanR2xS1InputsR2xS1Latent(SpecialEuclideanR2xS1InputsR2xS1Latent):
    """Separating invariances of (R2xS1)x(R2xS1)x(R2xS1) wrt SE(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of (R2xS1)x(R2xS1)x(R2xS1) wrt SE(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 12).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create reordering indices
        order = jnp.array([3, 4, 5, 0, 1, 2])

        # Use take to reorder along the last dimension
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


if __name__ == "__main__":
    # Set random key and dimensions
    key = jax.random.PRNGKey(0)
    B, S, N = 2, 5, 3  # Batch, sample pairs, latent points

    # For these tests, inputs are of shape (B, S, 2, 3)
    #   - For each sample, the first 2 entries are a 2D position and the third is an angle.
    inputs_R2 = jax.random.normal(key, (B, S, 2, 3))

    # Latent pose:
    # p[0] is positions: (B, N, 2)
    p_R2_pos = jax.random.normal(key, (B, N, 2))
    # p[1] is orientation: a 2-vector; we normalize to unit length.
    p_R2_ori = jax.random.normal(key, (B, N, 2))
    p_R2_ori = p_R2_ori / jnp.linalg.norm(p_R2_ori, axis=-1, keepdims=True)
    p_R2 = (p_R2_pos, p_R2_ori)

    # Test EuclideanR2xS1InputsR2xS1Latent
    model1 = EuclideanR2xS1InputsR2xS1Latent()
    variables1 = model1.init(key, inputs_R2, p_R2)
    out1 = model1.apply(variables1, inputs_R2, p_R2)
    print("EuclideanR2xS1InputsR2xS1Latent output shape:", out1.shape)
    assert out1.shape == (B, S, N, 14), f"Expected (B,S,N,14), got {out1.shape}"

    # Test SymEuclideanR2xS1InputsR2xS1Latent
    model2 = SymEuclideanR2xS1InputsR2xS1Latent()
    variables2 = model2.init(key, inputs_R2, p_R2)
    out2 = model2.apply(variables2, inputs_R2, p_R2)
    print("SymEuclideanR2xS1InputsR2xS1Latent first output shape:", out2[0].shape)
    print("SymEuclideanR2xS1InputsR2xS1Latent second output shape:", out2[1].shape)
    assert out2[0].shape == (B, S, N, 14), f"Expected (B,S,N,14), got {out2[0].shape}"
    assert out2[1].shape == (B, S, N, 14), f"Expected (B,S,N,14), got {out2[1].shape}"

    # Test SpecialEuclideanR2xS1InputsR2xS1Latent
    model3 = SpecialEuclideanR2xS1InputsR2xS1Latent()
    variables3 = model3.init(key, inputs_R2, p_R2)
    out3 = model3.apply(variables3, inputs_R2, p_R2)
    print("SpecialEuclideanR2xS1InputsR2xS1Latent output shape:", out3.shape)
    assert out3.shape == (B, S, N, 6), f"Expected (B,S,N,6), got {out3.shape}"

    # Test SymSpecialEuclideanR2xS1InputsR2xS1Latent
    model4 = SymSpecialEuclideanR2xS1InputsR2xS1Latent()
    variables4 = model4.init(key, inputs_R2, p_R2)
    out4 = model4.apply(variables4, inputs_R2, p_R2)
    print(
        "SymSpecialEuclideanR2xS1InputsR2xS1Latent first output shape:", out4[0].shape
    )
    print(
        "SymSpecialEuclideanR2xS1InputsR2xS1Latent second output shape:", out4[1].shape
    )
    assert out4[0].shape == (B, S, N, 6), f"Expected (B,S,N,6), got {out4[0].shape}"
    assert out4[1].shape == (B, S, N, 6), f"Expected (B,S,N,6), got {out4[1].shape}"

    print("All tests passed!")
