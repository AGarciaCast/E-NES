from torch.utils.data import DataLoader
import numpy as np
import jax.numpy as jnp
import os
from scipy.interpolate import RegularGridInterpolator

from experiments.fitting.datasets.base_coordinate_dataset import (
    collate_batch_with_index,
    CoordinateDataset,
)

import math

import torch


class PositionOrientationCoordinateDataset(CoordinateDataset):
    def __init__(
        self,
        base_dataset,
        n_coords,
        num_pairs,
        precomputed_dir,
        x_min=None,
        x_max=None,
        theta_range=[
            0.0,
            2.0 * math.pi,
        ],
        interpolation_mode="linear",
        save_data=True,
        force_recompute=False,
    ):
        """
        CoordinateDataset generates coordinate pairs and their interpolated values
        for a given base dataset. Optionally caches precomputed data for faster access.

        Args:
            base_dataset (Dataset): The base dataset of velocity fields or similar.
            n_coords (int): Total number of coordinates per sample.
            num_pairs (int): Number of coordinate pairs per batch.
            dim_signal (int): Dimensionality of the signal (e.g., 2 for 2D images).
            precomputed_dir (str): Directory for storing/retrieving precomputed data.
            x_min (list or None): Minimum coordinate values for each dimension.
            x_max (list or None): Maximum coordinate values for each dimension.
            interpolation_mode (str): Interpolation mode for grid sampling (default linear').
            save_data (bool): If True, load precomputed data into memory.
            force_recompute (bool): If True, force recomputation of precomputed data.
        """

        # Coordinate bounds
        x_min = x_min if x_min is not None else [-1.0, -1.0]

        x_max = x_max if x_max is not None else [1.0, 1.0]

        self.ambient_x_min = np.array(x_min + [theta_range[0]], dtype=np.float32)

        self.ambient_x_max = np.array(
            x_max + [theta_range[1]],
            dtype=np.float32,
        )

        self.theta_range = theta_range

        super().__init__(
            base_dataset=base_dataset,
            n_coords=n_coords,
            num_pairs=num_pairs,
            dim_signal=2,
            precomputed_dir=precomputed_dir,
            x_min=np.array(x_min, dtype=np.float32),
            x_max=np.array(x_max, dtype=np.float32),
            interpolation_mode=interpolation_mode,
            save_data=save_data,
            force_recompute=force_recompute,
        )

    def _compute_sample(self, idx):
        """
        Compute precomputed data for a single sample and save it to disk.
        """
        velocity, _ = self.base_dataset[idx]
        if isinstance(velocity, jnp.ndarray):
            velocity = np.array(velocity)  # Convert to numpy if it's a JAX array
        elif not isinstance(velocity, np.ndarray):
            velocity = velocity.numpy()  # For PyTorch tensors

        # Generate coordinate chunks
        coord_chunks = self._generate_coords()
        nx, ny, nz = velocity.shape
        x = np.linspace(self.x_min[0], self.x_max[0], nx)
        y = np.linspace(self.x_min[1], self.x_max[1], ny)
        z = np.linspace(
            self.theta_range[0], self.theta_range[1], nz, endpoint=False
        )  # z-axis (periodic)

        # Extend the data along the z-axis
        z_extended = np.concatenate([z, [self.theta_range[1]]])  # Append upper limit
        velocity_extended = np.concatenate(
            [velocity, velocity[:, :, :1]], axis=2
        )  # Duplicate z=0 slice at z=2*pi

        # Create interpolator
        interpolator = RegularGridInterpolator(
            (x, y, z_extended),
            velocity_extended,
            method=self.interpolation_mode,
            bounds_error=False,
            fill_value=None,
        )
        precomputed_chunks = []

        for chunk in coord_chunks:
            # Use NumPy arrays instead of immediately converting to JAX arrays
            # This is more efficient as JAX will convert them when needed
            interpolated_values = np.array(
                interpolator(chunk), dtype=np.float32
            ).reshape(self.num_pairs, 2)

            coords_pairs = np.array(chunk, dtype=np.float32).reshape(
                self.num_pairs, 2, 3
            )

            precomputed_chunks.append(
                {
                    "interpolated_values": interpolated_values,
                    "coords_pairs": coords_pairs,
                }
            )

        # Save to disk using NumPy
        np.save(
            os.path.join(self.precomputed_dir, f"sample_{idx}.npy"), precomputed_chunks
        )

    def _generate_coords(self):
        """Generate random coordinate chunks for a single sample."""
        coords = np.random.rand(self.n_coords, 3)  # Random in [0, 1]
        coords = (self.ambient_x_max - self.ambient_x_min) * coords + self.ambient_x_min

        # Ensure that you are in the domain
        coords[..., 2] = (coords[..., 2] - self.theta_range[0]) % (
            2.0 * math.pi
        ) + self.theta_range[0]

        np.random.shuffle(coords)
        return [
            coords[i : i + self.num_pairs * 2]
            for i in range(0, self.n_coords, self.num_pairs * 2)
        ]


def create_position_orientation_dataloader(
    base_dataset,
    batch_size,
    n_coords,
    num_pairs,
    precomputed_dir,
    shuffle=True,
    x_min=None,
    x_max=None,
    theta_range=[
        0.0,
        2.0 * math.pi,
    ],
    num_workers=0,
    pin_memory=True,
    save_data=True,
    force_recompute=False,
    persistent_workers=False,
    drop_last=False,
    seed=0,
):
    """
    Creates a DataLoader for the CoordinateDataset.

    Args:
        base_dataset (Dataset): The base dataset (e.g., velocity fields).
        batch_size (int): Number of samples per batch.
        n_coords (int): Total number of coordinates per image.
        num_pairs (int): Number of coordinate pairs per batch.
        dim_signal (int): Dimensionality of the signal (e.g., 2D or 3D).
        precomputed_dir (str): Directory for storing/retrieving precomputed data.
        x_min (list or None): Minimum coordinate values for each dimension.
        x_max (list or None): Maximum coordinate values for each dimension.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to use pinned memory.
        save_data (bool): If True, load precomputed data into memory.
        force_recompute (bool): If True, force recomputation of precomputed data.

    Returns:
        DataLoader: The DataLoader for the CoordinateDataset.
    """

    dataset = PositionOrientationCoordinateDataset(
        base_dataset=base_dataset,
        n_coords=n_coords,
        num_pairs=num_pairs,
        precomputed_dir=precomputed_dir,
        x_min=x_min,
        x_max=x_max,
        theta_range=theta_range,
        interpolation_mode="linear",
        save_data=save_data,
        force_recompute=force_recompute,
    )

    batch_size = min(len(dataset), batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch_with_index,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        drop_last=drop_last,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),  # Add this
        generator=torch.Generator().manual_seed(seed),
    )
    return dataloader
