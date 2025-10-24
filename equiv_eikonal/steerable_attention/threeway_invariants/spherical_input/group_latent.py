import jax
import jax.numpy as jnp

from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)


class BaseSphericalGroupLatent(BaseThreewayInvariants):
    """Separating invariances of RNxRNxG wrt G (with G an affine homogenous group)."""

    def setup(self):
        super().setup()

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of RNxRNxG wrt G (with G an affine homogenous group).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, n).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, n, n). Shape of second component
                None.
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 2n).
        """

        # Cast to double precision for more stable calculation
        p_ori, _ = p

        # Swap axes with stable handling of potentially transposed tensors
        p_ori = jnp.swapaxes(p_ori, -2, -1)

        # Extract x and y components
        x = inputs[:, :, 0, :]  # (B, S, n)
        y = inputs[:, :, 1, :]  # (B, S, n)

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

        # Apply the latent orientation with stable einsum
        iso_x = jnp.einsum("blij,bsj->bsli", p_ori, x)
        iso_y = jnp.einsum("blij,bsj->bsli", p_ori, y)

        #  Concatenate along the last axis
        separating_inv = jnp.concatenate([iso_x, iso_y], axis=-1)

        # Convert back to original precision
        return separating_inv


##############################################################################
### S1 ###
##############################################################################


class OrthogonalS1InputsO2Latent(BaseSphericalGroupLatent):
    """Separating invariances of S1xS1xO(2) wrt O(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S1xS1xO(2) wrt O(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 1).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2, 2). Shape of second component
                None.
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        return super().__call__(inputs, p)


class SymOrthogonalS1InputsO2Latent(OrthogonalS1InputsO2Latent):
    """Separating invariances of S1xS1xO(2) wrt O(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S1xS1xO(2) wrt O(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 1).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2, 2). Shape of second component
                None.
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialOrthogonalS1InputsSO2Latent(BaseSphericalGroupLatent):
    """Separating invariances of S1xS1xSO(2) wrt SO(2)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S1xS1xSO(2) wrt SO(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 1).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2, 2). Shape of second component
                None.
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        return super().__call__(inputs, p)


class SymSpecialOrthogonalS1InputsSO2Latent(SpecialOrthogonalS1InputsSO2Latent):
    """Separating invariances of S1xS1xSO(2) wrt SO(2) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S1xS1xSO(2) wrt SO(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 1).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2, 2). Shape of second component
                None.
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


##############################################################################
### S2 ###
##############################################################################


class OrthogonalS2InputsO3Latent(BaseSphericalGroupLatent):
    """Separating invariances of S2xS2xO(3) wrt O(3)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S2xS2xO(3) wrt O(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3, 3). Shape of second component
                None.
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        return super().__call__(inputs, p)


class SymOrthogonalS2InputsO3Latent(OrthogonalS2InputsO3Latent):
    """Separating invariances of S2xS2xO(3) wrt O(3) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S2xS2xO(3) wrt O(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3,3). Shape of second component
                None.
        Returns:
           invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([3, 4, 5, 0, 1, 2], dtype=jnp.int32)
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialOrthogonalS2InputsSO3Latent(BaseSphericalGroupLatent):
    """Separating invariances of S2xS2xSO(3) wrt SO(3)."""

    def setup(self):
        super().setup()
        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S2xS2xSO(3) wrt SO(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3, 3). Shape of second component
                None.
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """
        return super().__call__(inputs, p)


class SymSpecialOrthogonalS2InputsSO3Latent(SpecialOrthogonalS2InputsSO3Latent):
    """Separating invariances of S2xS2xSO(3) wrt SO(3) for both inputs (x,y) and (y,x)."""

    def setup(self):
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of S2xS2xSO(3) wrt SO(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, None]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3, 3). Shape of second component
                None.
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

    # ---- S1 Invariants ----
    # For S1: inputs shape (B, S, 2, 1)
    # Latent pose shape (B, N, 2, 2)
    inputs_S1 = jax.random.normal(key, (B, S, 2, 1))
    p_S1_pos = jax.random.normal(key, (B, N, 2, 2))
    p_S1 = (p_S1_pos, None)
    # OrthogonalS1InputsO2Latent (expected output shape: (B, S, N, 4))
    out_S1 = OrthogonalS1InputsO2Latent().init_with_output(key, inputs_S1, p_S1)[0]
    print("OrthogonalS1InputsO2Latent output shape:", out_S1.shape)
    assert out_S1.shape == (B, S, N, 4), f"Expected (B,S,N,4), got {out_S1.shape}"

    # SymOrthogonalS1InputsO2Latent (tuple with two outputs, each of shape: (B, S, N, 4))
    out_S1_sym = SymOrthogonalS1InputsO2Latent().init_with_output(key, inputs_S1, p_S1)[
        0
    ]
    print("SymOrthogonalS1InputsO2Latent first output shape:", out_S1_sym[0].shape)
    print("SymOrthogonalS1InputsO2Latent second output shape:", out_S1_sym[1].shape)
    assert out_S1_sym[0].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_S1_sym[0].shape}"
    assert out_S1_sym[1].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_S1_sym[1].shape}"

    # SpecialOrthogonalS1InputsSO2Latent (expected output shape: (B, S, N, 4))
    out_spec_S1 = SpecialOrthogonalS1InputsSO2Latent().init_with_output(
        key, inputs_S1, p_S1
    )[0]
    print("SpecialOrthogonalS1InputsSO2Latent output shape:", out_spec_S1.shape)
    assert out_spec_S1.shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_S1.shape}"

    # SymSpecialOrthogonalS1InputsSO2Latent (tuple with two outputs, each of shape: (B, S, N, 4))
    out_spec_S1_sym = SymSpecialOrthogonalS1InputsSO2Latent().init_with_output(
        key, inputs_S1, p_S1
    )[0]
    print(
        "SymSpecialOrthogonalS1InputsSO2Latent first output shape:",
        out_spec_S1_sym[0].shape,
    )
    print(
        "SymSpecialOrthogonalS1InputsSO2Latent second output shape:",
        out_spec_S1_sym[1].shape,
    )
    assert out_spec_S1_sym[0].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_S1_sym[0].shape}"
    assert out_spec_S1_sym[1].shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_S1_sym[1].shape}"

    # ---- S2 Invariants ----
    # For S2: inputs shape (B, S, 2, 3)
    # Latent pose shape (B, N, 3, 3)
    inputs_S2 = jax.random.normal(key, (B, S, 2, 2))
    p_S2_pos = jax.random.normal(key, (B, N, 3, 3))
    p_S2 = (p_S2_pos, None)

    # OrthogonalS2InputsO3Latent (expected output shape: (B, S, N, 6))
    out_S2 = OrthogonalS2InputsO3Latent().init_with_output(key, inputs_S2, p_S2)[0]
    print("OrthogonalS2InputsO3Latent output shape:", out_S2.shape)
    assert out_S2.shape == (B, S, N, 6), f"Expected (B,S,N,6), got {out_S2.shape}"

    # SymOrthogonalS2InputsO3Latent (tuple with two outputs, each of shape: (B, S, N, 6))
    out_S2_sym = SymOrthogonalS2InputsO3Latent().init_with_output(key, inputs_S2, p_S2)[
        0
    ]
    print("SymOrthogonalS2InputsO3Latent first output shape:", out_S2_sym[0].shape)
    print("SymOrthogonalS2InputsO3Latent second output shape:", out_S2_sym[1].shape)
    assert out_S2_sym[0].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_S2_sym[0].shape}"
    assert out_S2_sym[1].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_S2_sym[1].shape}"

    # SpecialOrthogonalS2InputsSO3Latent (expected output shape: (B, S, N, 6))
    out_spec_S2 = SpecialOrthogonalS2InputsSO3Latent().init_with_output(
        key, inputs_S2, p_S2
    )[0]
    print("SpecialOrthogonalS2InputsSO3Latent output shape:", out_spec_S2.shape)
    assert out_spec_S2.shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_S2.shape}"

    # SymSpecialOrthogonalS2InputsSO3Latent (tuple with two outputs, each of shape: (B, S, N, 6))
    out_spec_S2_sym = SymSpecialOrthogonalS2InputsSO3Latent().init_with_output(
        key, inputs_S2, p_S2
    )[0]
    print(
        "SpecialOrthogonalS2InputsSO3Latent first output shape:",
        out_spec_S2_sym[0].shape,
    )
    print(
        "SpecialOrthogonalS2InputsSO3Latent second output shape:",
        out_spec_S2_sym[1].shape,
    )
    assert out_spec_S2_sym[0].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_S2_sym[0].shape}"
    assert out_spec_S2_sym[1].shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_S2_sym[1].shape}"

    print("All tests passed!")
