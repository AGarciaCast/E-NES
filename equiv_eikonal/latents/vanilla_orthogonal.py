from equiv_eikonal.latents.utils import (
    init_positions_grid,
    init_appearances_mean,
    init_appearances_ones,
    init_orientations_random_uniform,
    soft_clip,
)

import math
from typing import Optional, List, Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as random

from flax import linen as nn


class VanillaOrthogonalLatents(nn.Module):
    num_signals: int
    num_latents: int
    dim_signal: int
    dim_orientation: int
    latent_dim: int
    norm_angles: bool = False

    def setup(self):
        assert (
            self.dim_signal == 2 or self.dim_signal == 3
        ), "Dim signal must be 2 or 3."

        self.appearance = self.param(
            "appearance",
            lambda key: init_appearances_ones(
                self.num_latents, self.num_signals, self.latent_dim
            ),
        )

        if self.dim_orientation == 1:
            self.pose_pos = self.param(
                "pose_pos",
                lambda key: init_orientations_random_uniform(
                    key,
                    self.num_latents,
                    self.num_signals,
                    self.dim_signal - 1,
                    norm=self.norm_angles,
                ),
            )
        elif self.dim_orientation <= self.dim_signal and self.dim_orientation > 0:
            self.pose_pos = self.param(
                "pose_pos",
                lambda key: init_orientations_random_uniform(
                    key,
                    self.num_latents,
                    self.num_signals,
                    int((self.dim_orientation * (self.dim_orientation - 1)) / 2),
                    norm=self.norm_angles,
                ),
            )
        else:
            raise ValueError("Invalid orientation dimension.")

    def __call__(self, idx):
        """Forward pass with index selection.

        Args:
            idx: Index of the signal to process

        Returns:
            Tuple of (pose_data, appearance_data)
        """
        # Get positions for the specified index
        pose_pos = jnp.take(self.pose_pos, idx, axis=0)
        pose_pos = self.angles_to_group(pose_pos)

        # Get appearance data
        appearance = jnp.take(self.appearance, idx, axis=0)

        return (pose_pos, None), appearance

    def angles_to_group(self, angles: jnp.ndarray) -> jnp.ndarray:
        """Convert angles to rotation matrices or vectors.

        Args:
            angles: Angular parameters

        Returns:
            Rotation matrices or vectors
        """
        if self.dim_signal == 2:
            theta = angles.squeeze(-1)
            theta = theta * jnp.pi if self.norm_angles else theta
            cos_t = jnp.cos(theta)
            sin_t = jnp.sin(theta)
            if self.dim_orientation == 2:
                pose_ori = jnp.stack(
                    [
                        jnp.stack([cos_t, -sin_t], axis=-1),
                        jnp.stack([sin_t, cos_t], axis=-1),
                    ],
                    axis=-2,
                )
            else:
                pose_ori = jnp.stack([cos_t, sin_t], axis=-1)
        else:
            theta = angles[..., 0]
            phi = angles[..., 1]

            # Scale angles if normalized
            if self.norm_angles:
                theta = theta * jnp.pi
                phi = phi * jnp.pi

            cos_t = jnp.cos(theta)
            sin_t = jnp.sin(theta)
            cos_p = jnp.cos(phi)
            sin_p = jnp.sin(phi)
            if self.dim_orientation == 1:
                pose_ori = jnp.stack([sin_t * cos_p, sin_t * sin_p, cos_t], axis=-1)
            elif self.dim_orientation == 2:
                theta = angles.squeeze(-1)
                theta = theta * jnp.pi if self.norm_angles else theta
                cos_t = jnp.cos(theta)
                sin_t = jnp.sin(theta)

                aux_ones = jnp.ones_like(theta)
                aux_zeros = jnp.zeros_like(theta)

                # Rotation matrix around z-axis (this is a design choice)
                pose_ori = jnp.stack(
                    [
                        jnp.stack([cos_t, -sin_t, aux_zeros], axis=-1),
                        jnp.stack([sin_t, cos_t, aux_zeros], axis=-1),
                        jnp.stack([aux_zeros, aux_zeros, aux_ones], axis=-1),
                    ],
                    axis=-2,
                )

            else:
                gamma = angles[..., 2]
                if self.norm_angles:
                    gamma = gamma * jnp.pi

                cos_g = jnp.cos(gamma)
                sin_g = jnp.sin(gamma)

                aux_ones = jnp.ones_like(gamma)
                aux_zeros = jnp.zeros_like(gamma)

                # Rotation matrices around x, y, z axes
                R_x = jnp.stack(
                    [
                        jnp.stack([aux_ones, aux_zeros, aux_zeros], axis=-1),
                        jnp.stack([aux_zeros, cos_t, -sin_t], axis=-1),
                        jnp.stack([aux_zeros, sin_t, cos_t], axis=-1),
                    ],
                    axis=-2,
                )

                R_y = jnp.stack(
                    [
                        jnp.stack([cos_p, aux_zeros, sin_p], axis=-1),
                        jnp.stack([aux_zeros, aux_ones, aux_zeros], axis=-1),
                        jnp.stack([-sin_p, aux_zeros, cos_p], axis=-1),
                    ],
                    axis=-2,
                )

                R_z = jnp.stack(
                    [
                        jnp.stack([cos_g, -sin_g, aux_zeros], axis=-1),
                        jnp.stack([sin_g, cos_g, aux_zeros], axis=-1),
                        jnp.stack([aux_zeros, aux_zeros, aux_ones], axis=-1),
                    ],
                    axis=-2,
                )

                # Compose rotation matrices: R = Rz @ Ry @ Rx
                pose_ori = jnp.matmul(R_z, jnp.matmul(R_y, R_x))

        return pose_ori

    def add_noise(self, autodecoder_params, noise_std: float, noise_key):
        """Add Gaussian noise to latent parameters.

        Args:
            autodecoder_params: Current parameters
            noise_std: Standard deviation of noise
            noise_key: Random key

        Returns:
            Updated parameters and new random key
        """
        # Split random key for position noise
        pos_key, new_key = jax.random.split(noise_key)

        # Generate noise for positions
        pose_pos_shape = autodecoder_params["params"]["pose_pos"].shape
        noise_pos = random.normal(pos_key, pose_pos_shape) * noise_std

        # Create new parameters with added noise
        new_params = jax.tree_map(lambda x: x, autodecoder_params)  # Make a copy
        new_params["params"]["pose_pos"] = (
            autodecoder_params["params"]["pose_pos"] + noise_pos
        )

        return new_params, new_key

    def clip_pos(self, autodecoder_params, vel_idx=None):
        """Clip positions and orientations to valid ranges.

        Args:
            autodecoder_params: Current parameters
            vel_idx: Optional index for partial updates

        Returns:
            Updated parameters with clipped values
        """
        # Create a copy of parameters
        new_params = jax.tree_map(lambda x: x, autodecoder_params)

        # Clip positions
        if vel_idx is not None:
            # Get current positions
            current_pos = new_params["params"]["pose_pos"][vel_idx]
            # Clip positions
            # clipped_pos = soft_clip(current_pos, self.xmin, self.xmax, alpha=5)
            clipped_pos = self._clip_orientation_values(current_pos)
            # Update specific positions
            new_params["params"]["pose_pos"] = (
                new_params["params"]["pose_pos"].at[vel_idx].set(clipped_pos)
            )
        else:
            # Get all positions
            current_pos = new_params["params"]["pose_pos"]
            # Clip positions
            clipped_pos = self._clip_orientation_values(current_pos)
            # Update all positions
            new_params["params"]["pose_pos"] = clipped_pos

        return new_params

    def _clip_orientation_values(self, ori):
        """Clip orientation values to valid ranges.

        Args:
            ori: Orientation values

        Returns:
            Clipped orientation values
        """
        # Full dimensional orientation
        if self.dim_orientation == 1:

            theta = ori[..., 0]
            phi = ori[..., 1]

            # Clip phi values
            if self.norm_angles:
                phi = jnp.remainder(phi + 1.0, 2.0) - 1.0
            else:
                phi = jnp.remainder(phi, 2.0 * jnp.pi)

            # Clip theta values (special handling)
            if self.norm_angles:
                # Make values modulo 2 and map values in [1, 2) to [-1, 0)
                t_mod = jnp.mod(theta, 2)
                theta = jnp.where(t_mod >= 1, t_mod - 2, t_mod)
            else:
                # Make values modulo 2π and map values in [π, 2π) to [0, π)
                t_mod = jnp.mod(theta, 2 * jnp.pi)
                theta = jnp.where(t_mod >= jnp.pi, 2 * jnp.pi - t_mod, t_mod)

            # Concatenate clipped values
            return jnp.stack([theta, phi], axis=-1)
        else:
            if self.norm_angles:
                return jnp.remainder(ori + 1.0, 2.0) - 1.0
            else:
                return jnp.remainder(ori, 2.0 * jnp.pi)
        # Partial dimensional orientation


class VanillaMetaOrthogonalLatents(VanillaOrthogonalLatents):
    def __call__(self):
        """Forward pass processing all signals.

        Returns:
            Tuple of (pose_data, appearance_data)
        """
        # Use all positions
        pose_pos = self.angles_to_group(self.pose_pos)

        # Use all appearance data
        appearance = self.appearance

        return (pose_pos, None), appearance
