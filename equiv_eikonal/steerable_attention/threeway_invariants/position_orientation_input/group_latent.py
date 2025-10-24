from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
import jax
import jax.numpy as jnp


class BasePositionOrientationGroupLatent(BaseThreewayInvariants):
    """Separating invariances of (R2xS1)x(R2xS1)xG wrt G (with G an affine homogenous group)."""

    def setup(self):
        super().setup()

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of (R2xS1)x(R2xS1)xG wrt G (with G an affine homogenous group).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 8).
        """
        p_pos, p_ori = p

        p_ori = jnp.transpose(p_ori, axes=(0, 1, 3, 2))
        x_pos = inputs[:, :, 0, :2]
        y_pos = inputs[:, :, 1, :2]
        x_theta = inputs[:, :, 0, 2]
        y_theta = inputs[:, :, 1, 2]

        cos_x_t = jnp.cos(x_theta)
        sin_x_t = jnp.sin(x_theta)

        # Create rotation matrices
        x_ori = jnp.stack(
            [
                jnp.stack([cos_x_t, -sin_x_t], axis=-1),
                jnp.stack([sin_x_t, cos_x_t], axis=-1),
            ],
            axis=-2,
        )

        cos_y_t = jnp.cos(y_theta)
        sin_y_t = jnp.sin(y_theta)

        # Create rotation matrices
        y_ori = jnp.stack(
            [
                jnp.stack([cos_y_t, -sin_y_t], axis=-1),
                jnp.stack([sin_y_t, cos_y_t], axis=-1),
            ],
            axis=-2,
        )

        # Broadcast x and y
        x_pos_bc = jnp.expand_dims(
            x_pos, axis=2
        )  # Shape: (batch_size, num_sample_pairs, 1, 2)
        y_pos_bc = jnp.expand_dims(
            y_pos, axis=2
        )  # Shape: (batch_size, num_sample_pairs, 1, 2)

        # Broadcast p_pos
        p_pos_bc = jnp.expand_dims(
            p_pos, axis=1
        )  # Shape: (batch_size, 1, num_latents, 2)

        # Computing invariants
        x_lat_diff = x_pos_bc - p_pos_bc
        y_lat_diff = y_pos_bc - p_pos_bc

        # Using einsum for the matrix multiplication
        iso_x_pos = jnp.einsum("blij,bslj->bsli", p_ori, x_lat_diff)
        iso_y_pos = jnp.einsum("blij,bslj->bsli", p_ori, y_lat_diff)

        # Calculate orientation invariants
        iso_x_ori = jnp.einsum("blij,bsjk->bslik", p_ori, x_ori)[:, :, :, :, 0]
        iso_y_ori = jnp.einsum("blij,bsjk->bslik", p_ori, y_ori)[:, :, :, :, 0]

        # Concatenate all invariants
        separating_inv = jnp.concatenate(
            [iso_x_pos, iso_x_ori, iso_y_pos, iso_y_ori],
            axis=-1,
        )

        return separating_inv


class EuclideanR2xS1InputsE2Latent(BasePositionOrientationGroupLatent):
    """Separating invariances of (R2xS1)x(R2xS1)xE(2) wrt E(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant
        self.dim = 8

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of (R2xS1)x(R2xS1)xE(2) wrt E(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 8).
        """
        return super().__call__(inputs, p)


class SymEuclideanR2xS1InputsE2Latent(EuclideanR2xS1InputsE2Latent):
    """Separating invariances of (R2xS1)x(R2xS1)xE(2) wrt E(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of (R2xS1)x(R2xS1)xE(2) wrt E(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 16).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create index array for reordering
        order = jnp.array([4, 5, 6, 7, 0, 1, 2, 3])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR2xS1InputsSE2Latent(BasePositionOrientationGroupLatent):
    """Separating invariances of (R2xS1)x(R2xS1)xSE(2) wrt SE(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant
        self.dim = 8

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xSE(2) wrt SE(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 8).
        """
        return super().__call__(inputs, p)


class SymSpecialEuclideanR2xS1InputsSE2Latent(SpecialEuclideanR2xS1InputsSE2Latent):
    """Separating invariances of (R2xS1)x(R2xS1)xSE(2) wrt SE(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of (R2xS1)x(R2xS1)xSE(2) wrt SE(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 16).
        """
        separating_inv_xy = super().__call__(inputs, p)

        # Create index array for reordering
        order = jnp.array([4, 5, 6, 7, 0, 1, 2, 3])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, S, N = 2, 5, 3  # Batch size, number of sample pairs, number of latents

    # -------------------------------------------------------------------------
    # Tests for the R2×S1 functions (position+orientation in R2×S1)
    # -------------------------------------------------------------------------
    # Inputs: shape (B, S, 2, 3)
    #   Here the last dimension: first 2 entries are a 2D position, third entry is an angle.
    inputs_R2 = jax.random.normal(key, (B, S, 2, 3))
    # Latent pose p:
    #   p[0]: positions in R2, shape (B, N, 2)
    #   p[1]: orientation matrices in SO(2), shape (B, N, 2, 2)
    p_R2_pos = jax.random.normal(key, (B, N, 2))
    p_R2_ori = jax.random.normal(key, (B, N, 2, 2))
    p_R2 = (p_R2_pos, p_R2_ori)

    # BasePositionOrientationGroupLatent (directly callable)
    base_PO = BasePositionOrientationGroupLatent()
    out_base = base_PO(inputs_R2, p_R2)
    print("BasePositionOrientationGroupLatent output shape:", out_base.shape)
    assert out_base.shape == (B, S, N, 8), f"Expected (B,S,N,8), got {out_base.shape}"

    # EuclideanR2xS1InputsE2Latent: expected to return invariants of shape (B, S, N, 8)
    euc_R2 = EuclideanR2xS1InputsE2Latent()
    out_euc_R2 = euc_R2(inputs_R2, p_R2)
    print("EuclideanR2xS1InputsE2Latent output shape:", out_euc_R2.shape)
    assert out_euc_R2.shape == (
        B,
        S,
        N,
        8,
    ), f"Expected (B,S,N,8), got {out_euc_R2.shape}"

    # SymEuclideanR2xS1InputsE2Latent: returns a tuple (xy, yx), each of shape (B, S, N, 8)
    sym_euc_R2 = SymEuclideanR2xS1InputsE2Latent()
    out_sym_euc_R2 = sym_euc_R2(inputs_R2, p_R2)
    print(
        "SymEuclideanR2xS1InputsE2Latent first output shape:", out_sym_euc_R2[0].shape
    )
    print(
        "SymEuclideanR2xS1InputsE2Latent second output shape:", out_sym_euc_R2[1].shape
    )
    assert out_sym_euc_R2[0].shape == (
        B,
        S,
        N,
        8,
    ), f"Expected (B,S,N,8), got {out_sym_euc_R2[0].shape}"
    assert out_sym_euc_R2[1].shape == (
        B,
        S,
        N,
        8,
    ), f"Expected (B,S,N,8), got {out_sym_euc_R2[1].shape}"

    # SpecialEuclideanR2xS1InputsSE2Latent: expected shape (B, S, N, 8)
    spec_euc_R2 = SpecialEuclideanR2xS1InputsSE2Latent()
    out_spec_euc_R2 = spec_euc_R2(inputs_R2, p_R2)
    print("SpecialEuclideanR2xS1InputsSE2Latent output shape:", out_spec_euc_R2.shape)
    assert out_spec_euc_R2.shape == (
        B,
        S,
        N,
        8,
    ), f"Expected (B,S,N,8), got {out_spec_euc_R2.shape}"

    # SymSpecialEuclideanR2xS1InputsSE2Latent: returns a tuple of two arrays, each shape (B, S, N, 8)
    sym_spec_euc_R2 = SymSpecialEuclideanR2xS1InputsSE2Latent()
    out_sym_spec_euc_R2 = sym_spec_euc_R2(inputs_R2, p_R2)
    print(
        "SymSpecialEuclideanR2xS1InputsSE2Latent first output shape:",
        out_sym_spec_euc_R2[0].shape,
    )
    print(
        "SymSpecialEuclideanR2xS1InputsSE2Latent second output shape:",
        out_sym_spec_euc_R2[1].shape,
    )
    assert out_sym_spec_euc_R2[0].shape == (
        B,
        S,
        N,
        8,
    ), f"Expected (B,S,N,8), got {out_sym_spec_euc_R2[0].shape}"
    assert out_sym_spec_euc_R2[1].shape == (
        B,
        S,
        N,
        8,
    ), f"Expected (B,S,N,8), got {out_sym_spec_euc_R2[1].shape}"

    print("All tests passed!")
