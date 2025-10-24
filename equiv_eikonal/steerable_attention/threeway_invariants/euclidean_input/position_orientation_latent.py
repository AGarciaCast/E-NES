from equiv_eikonal.steerable_attention.threeway_invariants._base_invariant import (
    BaseThreewayInvariants,
)
import jax
import jax.numpy as jnp

##############################################################################
### R2 ###
##############################################################################


class EuclideanR2InputsR2xS1Latent(BaseThreewayInvariants):

    def super(self):
        """Separating invariances of R2xR2x(R2xS1) wrt E(2)."""
        super().setup()

        # Register the dimensionality of the invariant.
        self.dim = 5

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2x(R2xS1) wrt E(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 5).
        """

        p_pos, p_ori = p
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]
        y_bc = y[:, :, None, :]

        # Broadcast p_pos
        p_pos_bc = p_pos[:, None, :, :]

        # Reduction to the isotropy
        x_lat_diff = x_bc - p_pos_bc
        y_lat_diff = y_bc - p_pos_bc

        # In JAX, we need to be explicit about stacking and transposing
        A = jnp.stack(
            [p_ori, jnp.stack([-1.0 * p_ori[..., 1], p_ori[..., 0]], axis=-1)], axis=-1
        )
        A = jnp.swapaxes(A, -2, -1)

        # Replace torch.einsum with jax.numpy.einsum
        iso_x = jnp.einsum("blij,bslj->bsli", A, x_lat_diff)
        iso_y = jnp.einsum("blij,bslj->bsli", A, y_lat_diff)

        # Compute the invariants of R2xR2 wrt O(1)
        invariant1 = iso_x[..., 0]
        invariant2 = (iso_x[..., 1]) ** 2
        invariant3 = iso_y[..., 0]
        invariant4 = (iso_y[..., 1]) ** 2
        invariant5 = (iso_x[..., 1] - iso_y[..., 1]) ** 2

        separating_inv = jnp.stack(
            [invariant1, invariant2, invariant3, invariant4, invariant5], axis=-1
        )

        return separating_inv


class SymEuclideanR2InputsR2xS1Latent(EuclideanR2InputsR2xS1Latent):
    def super(self):
        """Separating invariances of R2xR2x(R2xS1) wrt E(2) for both inputs (x,y) and (y,x)."""
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2x(R2xS1) wrt E(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 5).
        """

        separating_inv_xy = super().__call__(inputs, p)

        # Use JAX's take instead of torch's index_select
        order = jnp.array([2, 3, 0, 1, 4])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR2InputsR2xS1Latent(BaseThreewayInvariants):

    def super(self):
        """Separating invariances of R2xR2x(R2xS1) wrt SE(2)."""
        super().setup()

        # Register the dimensionality of the invariant.
        self.dim = 4

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2x(R2xS1) wrt SE(2).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """

        p_pos, p_ori = p
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]
        y_bc = y[:, :, None, :]

        # Broadcast p_pos
        p_pos_bc = p_pos[:, None, :, :]

        x_lat_diff = x_bc - p_pos_bc
        y_lat_diff = y_bc - p_pos_bc

        A = jnp.stack(
            [p_ori, jnp.stack([-1.0 * p_ori[..., 1], p_ori[..., 0]], axis=-1)], axis=-1
        )
        A = jnp.swapaxes(A, -2, -1)

        # Notice that det(A) is always >0
        iso_x = jnp.einsum("blij,bslj->bsli", A, x_lat_diff)
        iso_y = jnp.einsum("blij,bslj->bsli", A, y_lat_diff)

        # Using concatenate instead of concat in JAX
        separating_inv = jnp.concatenate(
            [
                iso_x,
                iso_y,
            ],
            axis=-1,
        )

        return separating_inv


class SymSpecialEuclideanR2InputsR2xS1Latent(SpecialEuclideanR2InputsR2xS1Latent):
    def super(self):
        """Separating invariances of R2xR2x(R2xS1) wrt SE(2) for both inputs (x,y) and (y,x)."""
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R2xR2x(R2xS1) wrt SE(2) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 2).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 2). Shape of second component
                (batch_size, num_latents, 2).
        Returns:
            invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 4).
        """

        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


##############################################################################
### R3 ###
##############################################################################


class EuclideanR3InputsR3xS2Latent(BaseThreewayInvariants):

    def super(self):
        """Separating invariances of R3xR3x(R3xS2) wrt E(3)."""
        super().setup()

        # Register the dimensionality of the invariant.
        self.dim = 5

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3x(R3xS2) wrt E(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 5).
        """

        p_pos, p_ori = p
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]
        y_bc = y[:, :, None, :]

        # Broadcast p_pos
        p_pos_bc = p_pos[:, None, :, :]

        # Reduction to the isotropy
        x_lat_diff = x_bc - p_pos_bc
        y_lat_diff = y_bc - p_pos_bc

        # Deterministic approach to create orthogonal vectors
        # Find the index of the smallest absolute component of p_ori
        axis_dim = p_ori.shape[-1]
        abs_p_ori = jnp.abs(p_ori)
        mask = jnp.eye(axis_dim)[jnp.argmin(abs_p_ori, axis=-1)]

        # Create a deterministic vector perpendicular to p_ori
        # This is equivalent to choosing (1,0,0), (0,1,0), or (0,0,1) based on which component
        # of p_ori has the smallest absolute value, then making it perpendicular
        v = mask - (jnp.sum(mask * p_ori, axis=-1, keepdims=True) * p_ori)
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)  # Normalize

        # First orthogonal vector
        u1 = jnp.cross(p_ori, v, axis=-1)
        u1 = u1 / jnp.linalg.norm(u1, axis=-1, keepdims=True)  # Normalize

        # Second orthogonal vector
        u2 = jnp.cross(p_ori, u1, axis=-1)

        # Concatenate to form the rotation matrix
        A = jnp.stack([p_ori, u1, u2], axis=-1)
        A = jnp.swapaxes(A, -2, -1)

        iso_x = jnp.einsum("blij,bslj->bsli", A, x_lat_diff)
        iso_y = jnp.einsum("blij,bslj->bsli", A, y_lat_diff)

        # Compute the invariants of R3xR3 wrt O(2)
        iso_x12 = iso_x[..., 1:]
        iso_y12 = iso_y[..., 1:]

        invariant1 = iso_x[..., 0]
        invariant2 = jnp.sum(iso_x12**2, axis=-1)
        invariant3 = iso_y[..., 0]
        invariant4 = jnp.sum(iso_y12**2, axis=-1)
        invariant5 = jnp.sum((iso_x12 - iso_y12) ** 2, axis=-1)

        separating_inv = jnp.stack(
            [invariant1, invariant2, invariant3, invariant4, invariant5], axis=-1
        )
        return separating_inv


class SymEuclideanR3InputsR3xS2Latent(EuclideanR3InputsR3xS2Latent):
    def super(self):
        """Separating invariances of R3xR3x(R3xS2) wrt E(3) for both inputs (x,y) and (y,x)."""
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3x(R3xS2) wrt E(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3).
        Returns:
            invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 5).
        """

        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1, 4])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)

        return (separating_inv_xy, separating_inv_yx)


class SpecialEuclideanR3InputsR3xS2Latent(BaseThreewayInvariants):

    def super(self):
        """Separating invariances of R3xR3x(R3xS2) wrt SE(3)."""
        super().setup()

        # Register the dimensionality of the invariant.
        self.dim = 6

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3x(R3xS2) wrt SE(3).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3).
        Returns:
            invariants (jnp.ndarray): Separating invariances.
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """

        p_pos, p_ori = p
        x = inputs[:, :, 0, :]
        y = inputs[:, :, 1, :]

        # Broadcast x and y
        x_bc = x[:, :, None, :]
        y_bc = y[:, :, None, :]

        # Broadcast p_pos
        p_pos_bc = p_pos[:, None, :, :]

        # Reduction to the isotropy
        x_lat_diff = x_bc - p_pos_bc
        y_lat_diff = y_bc - p_pos_bc

        # Deterministic approach to create orthogonal vectors
        # Find the index of the smallest absolute component of p_ori
        axis_dim = p_ori.shape[-1]
        abs_p_ori = jnp.abs(p_ori)
        mask = jnp.eye(axis_dim)[jnp.argmin(abs_p_ori, axis=-1)]

        # Create a deterministic vector perpendicular to p_ori
        # This is equivalent to choosing (1,0,0), (0,1,0), or (0,0,1) based on which component
        # of p_ori has the smallest absolute value, then making it perpendicular
        v = mask - (jnp.sum(mask * p_ori, axis=-1, keepdims=True) * p_ori)
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)  # Normalize

        # First orthogonal vector
        u1 = jnp.cross(p_ori, v, axis=-1)
        u1 = u1 / jnp.linalg.norm(u1, axis=-1, keepdims=True)  # Normalize

        # Second orthogonal vector
        u2 = jnp.cross(p_ori, u1, axis=-1)

        # Concatenate to form the rotation matrix
        A = jnp.stack([p_ori, u1, u2], axis=-1)
        A = jnp.swapaxes(A, -2, -1)

        # Notice that det(A) is always >0
        iso_x = jnp.einsum("blij,bslj->bsli", A, x_lat_diff)
        iso_y = jnp.einsum("blij,bslj->bsli", A, y_lat_diff)

        # Compute the invariants of R3xR3 wrt SO(2)
        iso_x12 = iso_x[..., 1:]
        iso_y12 = iso_y[..., 1:]

        invariant1 = iso_x[..., 0]

        invariant2 = jnp.sum(iso_x12**2, axis=-1)

        invariant3 = iso_y[..., 0]

        invariant4 = jnp.sum(iso_y12**2, axis=-1)

        invariant5 = jnp.linalg.norm((iso_x12 - iso_y12) ** 2, axis=-1)

        # Using JAX's determinant function
        invariant6 = jnp.linalg.det(jnp.stack([iso_x12, iso_y12], axis=-1))

        separating_inv = jnp.stack(
            [invariant1, invariant2, invariant3, invariant4, invariant5, invariant6],
            axis=-1,
        )
        return separating_inv


class SymSpecialEuclideanR3InputsR3xS2Latent(SpecialEuclideanR3InputsR3xS2Latent):
    def super(self):
        """Separating invariances of R3xR3x(R3xS2) wrt SE(3) for both inputs (x,y) and (y,x)."""
        super().setup()
        self.symmetric = True

    def __call__(self, inputs, p):
        """Calculate the set of separating invariances of R3xR3x(R3xS2) wrt SE(3) for (x,y) and (y,x).

        Args:
            inputs (jnp.ndarray): The position of the input points. Shape (batch_size, num_sample_pairs, 2, 3).
            p (Tuple[jnp.ndarray, jnp.ndarray]): The pose of the latent points. Shape of first component
                (batch_size, num_latents, 3). Shape of second component
                (batch_size, num_latents, 3).
        Returns:
            invariants (Tuple[jnp.ndarray, jnp.ndarray]): Separating invariances for (x,y) and (y,x).
                Shape (batch_size, num_sample_pairs, num_latents, 6).
        """

        separating_inv_xy = super().__call__(inputs, p)

        order = jnp.array([2, 3, 0, 1, 4, 5])
        separating_inv_yx = jnp.take(separating_inv_xy, order, axis=-1)
        separating_inv_yx = separating_inv_yx.at[..., 5].multiply(-1.0)

        return (separating_inv_xy, separating_inv_yx)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, S, N = 2, 5, 3

    # ---- R2xS1 Invariants ----
    # For R2: inputs shape (B, S, 2, 2)
    # Latent pose: p[0] shape (B, N, 2), p[1] shape (B, N, 2)
    inputs_R2 = jax.random.normal(key, (B, S, 2, 2))
    p_R2_pos = jax.random.normal(key, (B, N, 2))
    p_R2_ori = jax.random.normal(key, (B, N, 2))
    p_R2 = (p_R2_pos, p_R2_ori)

    # EuclideanR2InputsR2xS1Latent: expected output shape (B, S, N, 5)
    out_R2 = EuclideanR2InputsR2xS1Latent().init_with_output(key, inputs_R2, p_R2)[0]
    print("EuclideanR2InputsR2xS1Latent output shape:", out_R2.shape)
    assert out_R2.shape == (B, S, N, 5), f"Expected (B,S,N,5), got {out_R2.shape}"

    # SymEuclideanR2InputsR2xS1Latent: returns tuple, each of shape (B, S, N, 5)
    out_R2_sym = SymEuclideanR2InputsR2xS1Latent().init_with_output(
        key, inputs_R2, p_R2
    )[0]
    print("SymEuclideanR2InputsR2xS1Latent first output shape:", out_R2_sym[0].shape)
    print("SymEuclideanR2InputsR2xS1Latent second output shape:", out_R2_sym[1].shape)
    assert out_R2_sym[0].shape == (
        B,
        S,
        N,
        5,
    ), f"Expected (B,S,N,5), got {out_R2_sym[0].shape}"
    assert out_R2_sym[1].shape == (
        B,
        S,
        N,
        5,
    ), f"Expected (B,S,N,5), got {out_R2_sym[1].shape}"

    # SpecialEuclideanR2InputsR2xS1Latent: expected output shape (B, S, N, 4)
    out_spec_R2 = SpecialEuclideanR2InputsR2xS1Latent().init_with_output(
        key, inputs_R2, p_R2
    )[0]
    print("SpecialEuclideanR2InputsR2xS1Latent output shape:", out_spec_R2.shape)
    assert out_spec_R2.shape == (
        B,
        S,
        N,
        4,
    ), f"Expected (B,S,N,4), got {out_spec_R2.shape}"

    # SymSpecialEuclideanR2InputsR2xS1Latent: returns tuple, each of shape (B, S, N, 4)
    out_spec_R2_sym = SymSpecialEuclideanR2InputsR2xS1Latent().init_with_output(
        key, inputs_R2, p_R2
    )[0]
    print(
        "SymSpecialEuclideanR2InputsR2xS1Latent first output shape:",
        out_spec_R2_sym[0].shape,
    )
    print(
        "SymSpecialEuclideanR2InputsR2xS1Latent second output shape:",
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

    # ---- R3xS2 Invariants ----
    # For R3: inputs shape (B, S, 2, 3)
    # Latent pose: p[0] shape (B, N, 3), p[1] shape (B, N, 3) representing an S2 element.
    inputs_R3 = jax.random.normal(key, (B, S, 2, 3))
    p_R3_pos = jax.random.normal(key, (B, N, 3))
    p_R3_ori = jax.random.normal(key, (B, N, 3))
    p_R3 = (p_R3_pos, p_R3_ori)

    # EuclideanR3InputsR3xS2Latent: expected shape (B, S, N, 5)
    out_R3 = EuclideanR3InputsR3xS2Latent().init_with_output(key, inputs_R3, p_R3)[0]
    print("EuclideanR3InputsR3xS2Latent output shape:", out_R3.shape)
    assert out_R3.shape == (B, S, N, 5), f"Expected (B,S,N,5), got {out_R3.shape}"

    # SymEuclideanR3InputsR3xS2Latent: returns tuple, each shape (B, S, N, 5)
    out_R3_sym = SymEuclideanR3InputsR3xS2Latent().init_with_output(
        key, inputs_R3, p_R3
    )[0]
    print("SymEuclideanR3InputsR3xS2Latent first output shape:", out_R3_sym[0].shape)
    print("SymEuclideanR3InputsR3xS2Latent second output shape:", out_R3_sym[1].shape)
    assert out_R3_sym[0].shape == (
        B,
        S,
        N,
        5,
    ), f"Expected (B,S,N,5), got {out_R3_sym[0].shape}"
    assert out_R3_sym[1].shape == (
        B,
        S,
        N,
        5,
    ), f"Expected (B,S,N,5), got {out_R3_sym[1].shape}"

    # SpecialEuclideanR3InputsR3xS2Latent: expected shape (B, S, N, 6)
    out_spec_R3 = SpecialEuclideanR3InputsR3xS2Latent().init_with_output(
        key, inputs_R3, p_R3
    )[0]
    print("SpecialEuclideanR3InputsR3xS2Latent output shape:", out_spec_R3.shape)
    assert out_spec_R3.shape == (
        B,
        S,
        N,
        6,
    ), f"Expected (B,S,N,6), got {out_spec_R3.shape}"

    # SymSpecialEuclideanR3InputsR3xS2Latent: returns tuple, each shape (B, S, N, 6)
    out_spec_R3_sym = SymSpecialEuclideanR3InputsR3xS2Latent().init_with_output(
        key, inputs_R3, p_R3
    )[0]
    print(
        "SymSpecialEuclideanR3InputsR3xS2Latent first output shape:",
        out_spec_R3_sym[0].shape,
    )
    print(
        "SymSpecialEuclideanR3InputsR3xS2Latent second output shape:",
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
