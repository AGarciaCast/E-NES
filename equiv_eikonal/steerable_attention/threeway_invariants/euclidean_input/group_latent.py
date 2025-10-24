import jax
import jax.numpy as jnp

from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)


class BaseEuclideanGroupLatent(BaseThreewayInvariants):
    """Separating invariances of RNxRNxG wrt G (with G an affine homogenous group)."""

    def setup(self):
        super().setup()

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of RNxRNxG wrt G (with G an affine homogenous group).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, n).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, n). Shape of second component
                (batch_size, num_latents, n, n).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 2n).
        """

        # Cast to double precision for more stable calculation
        p_pos, p_ori = p

        # Swap axes with stable handling of potentially transposed tensors
        p_ori = jnp.swapaxes(p_ori, -2, -1)

        # Extract x and y components
        x = inputs[:, :, 0, :]  # (B, S, n)
        y = inputs[:, :, 1, :]  # (B, S, n)

        # Broadcast with explicit shapes for better numerical control
        x_bc = jnp.expand_dims(x, axis=2)  # (B, S, 1, n)
        y_bc = jnp.expand_dims(y, axis=2)  # (B, S, 1, n)
        p_pos_bc = jnp.expand_dims(p_pos, axis=1)  # (B, 1, N, n)

        # Compute differences with stable broadcasting
        x_lat_diff = x_bc - p_pos_bc  # (B, S, N, n)
        y_lat_diff = y_bc - p_pos_bc  # (B, S, N, n)

        # Apply the latent orientation with stable einsum
        iso_x = jnp.einsum("blij,bslj->bsli", p_ori, x_lat_diff)
        iso_y = jnp.einsum("blij,bslj->bsli", p_ori, y_lat_diff)

        #  Concatenate along the last axis
        separating_inv = jnp.concatenate([iso_x, iso_y], axis=-1)

        # Convert back to original precision
        return separating_inv


##############################################################################
### R2 ###
##############################################################################


class EuclideanR2InputsE2Latent(BaseEuclideanGroupLatent):
    """Separating invariances of R2xR2xE(2) wrt E(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xE(2) wrt E(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        return super().__call__(inputs, p)


class SymEuclideanR2InputsE2Latent(EuclideanR2InputsE2Latent):
    """Separating invariances of R2xR2xE(2) wrt E(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xE(2) wrt E(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR2InputsSE2Latent(BaseEuclideanGroupLatent):
    """Separating invariances of R2xR2xSE(2) wrt SE(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xSE(2) wrt SE(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        return super().__call__(inputs, p)


class SymSpecialEuclideanR2InputsSE2Latent(SpecialEuclideanR2InputsSE2Latent):
    """Separating invariances of R2xR2xSE(2) wrt SE(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2xSE(2) wrt SE(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2, 2).
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


##############################################################################
### R3 ###
##############################################################################


class EuclideanR3InputsE3Latent(BaseEuclideanGroupLatent):
    """Separating invariances of R3xR3xE(3) wrt E(3)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xE(3) wrt E(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        return super().__call__(inputs, p)


class SymEuclideanR3InputsE3Latent(EuclideanR3InputsE3Latent):
    """Separating invariances of R3xR3xE(3) wrt E(3) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xE(3) wrt E(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, 3).
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([3, 4, 5, 0, 1, 2], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR3InputsSE3Latent(BaseEuclideanGroupLatent):
    """Separating invariances of R3xR3xSE(3) wrt SE(3)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xSE(3) wrt SE(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        return super().__call__(inputs, p)


class SymSpecialEuclideanR3InputsSE3Latent(SpecialEuclideanR3InputsSE3Latent):
    """Separating invariances of R3xR3xSE(3) wrt SE(3) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3xSE(3) wrt SE(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3, 3).
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([3, 4, 5, 0, 1, 2], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


##############################################################################
### Main: Testing all functions using init_with_output
##############################################################################

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, S, N = 2, 5, 3  # Batch size, number of sample pairs, number of latents

    # ---- R2 Invariants ----
    # For R2: inputs shape (B, S, 2, 2)
    # Latent pose: positions shape (B, N, 2) and orientations shape (B, N, 2, 2)
    inputs_R2 = jax.random.normal(key, (B, S, 2, 2))
    p_R2_pos = jax.random.normal(key, (B, N, 2))
    p_R2_ori = jax.random.normal(key, (B, N, 2, 2))
    p_R2 = (p_R2_pos, p_R2_ori)

    # EuclideanR2InputsE2Latent (expected output shape: (B, S, N, 4))
    out_R2 = EuclideanR2InputsE2Latent().init_with_output(key, inputs_R2, p_R2)[0]
    print("EuclideanR2InputsE2Latent output shape:", out_R2.shape)
    assert out_R2.shape == (B, S, N, 4), f"Expected (B,S,N,4), got {out_R2.shape}"

    # SymEuclideanR2InputsE2Latent (tuple with two outputs, each of shape: (B, S, N, 4))
    out_R2_sym = SymEuclideanR2InputsE2Latent().init_with_output(key, inputs_R2, p_R2)[
        0
    ]
    print("SymEuclideanR2InputsE2Latent first output shape:", out_R2_sym[0].shape)
    print("SymEuclideanR2InputsE2Latent second output shape:", out_R2_sym[1].shape)
    assert out_R2_sym[0].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_R2_sym[0].shape}"
    assert out_R2_sym[1].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_R2_sym[1].shape}"

    # SpecialEuclideanR2InputsSE2Latent (expected output shape: (B, S, N, 4))
    out_spec_R2 = SpecialEuclideanR2InputsSE2Latent().init_with_output(
        key, inputs_R2, p_R2
    )[0]
    print("SpecialEuclideanR2InputsSE2Latent output shape:", out_spec_R2.shape)
    assert out_spec_R2.shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_R2.shape}"

    # SymSpecialEuclideanR2InputsSE2Latent (tuple with two outputs, each of shape: (B, S, N, 4))
    out_spec_R2_sym = SymSpecialEuclideanR2InputsSE2Latent().init_with_output(
        key, inputs_R2, p_R2
    )[0]
    print(
        "SymSpecialEuclideanR2InputsSE2Latent first output shape:",
        out_spec_R2_sym[0].shape,
    )
    print(
        "SymSpecialEuclideanR2InputsSE2Latent second output shape:",
        out_spec_R2_sym[1].shape,
    )
    assert out_spec_R2_sym[0].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_R2_sym[0].shape}"
    assert out_spec_R2_sym[1].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_R2_sym[1].shape}"

    # ---- R3 Invariants ----
    # For R3: inputs shape (B, S, 2, 3)
    # Latent pose: positions shape (B, N, 3) and orientations shape (B, N, 3, 3)
    inputs_R3 = jax.random.normal(key, (B, S, 2, 3))
    p_R3_pos = jax.random.normal(key, (B, N, 3))
    p_R3_ori = jax.random.normal(key, (B, N, 3, 3))
    p_R3 = (p_R3_pos, p_R3_ori)

    # EuclideanR3InputsE3Latent (expected output shape: (B, S, N, 6))
    out_R3 = EuclideanR3InputsE3Latent().init_with_output(key, inputs_R3, p_R3)[0]
    print("EuclideanR3InputsE3Latent output shape:", out_R3.shape)
    assert out_R3.shape == (B, S, N, 6), f"Expected (B,S,N,6), got {out_R3.shape}"

    # SymEuclideanR3InputsE3Latent (tuple with two outputs, each of shape: (B, S, N, 6))
    out_R3_sym = SymEuclideanR3InputsE3Latent().init_with_output(key, inputs_R3, p_R3)[
        0
    ]
    print("SymEuclideanR3InputsE3Latent first output shape:", out_R3_sym[0].shape)
    print("SymEuclideanR3InputsE3Latent second output shape:", out_R3_sym[1].shape)
    assert out_R3_sym[0].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_R3_sym[0].shape}"
    assert out_R3_sym[1].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_R3_sym[1].shape}"

    # SpecialEuclideanR3InputsSE3Latent (expected output shape: (B, S, N, 6))
    out_spec_R3 = SpecialEuclideanR3InputsSE3Latent().init_with_output(
        key, inputs_R3, p_R3
    )[0]
    print("SpecialEuclideanR3InputsSE3Latent output shape:", out_spec_R3.shape)
    assert out_spec_R3.shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_R3.shape}"

    # SymSpecialEuclideanR3InputsSE3Latent (tuple with two outputs, each of shape: (B, S, N, 6))
    out_spec_R3_sym = SymSpecialEuclideanR3InputsSE3Latent().init_with_output(
        key, inputs_R3, p_R3
    )[0]
    print(
        "SymSpecialEuclideanR3InputsSE3Latent first output shape:",
        out_spec_R3_sym[0].shape,
    )
    print(
        "SymSpecialEuclideanR3InputsSE3Latent second output shape:",
        out_spec_R3_sym[1].shape,
    )
    assert out_spec_R3_sym[0].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_R3_sym[0].shape}"
    assert out_spec_R3_sym[1].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_R3_sym[1].shape}"

    print("All tests passed!")
