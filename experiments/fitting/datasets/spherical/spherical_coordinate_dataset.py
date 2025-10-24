from torch.utils.data import DataLoader
import numpy as np
import jax.numpy as jnp
import os


from experiments.fitting.datasets.base_coordinate_dataset import (
    collate_batch_with_index,
    CoordinateDataset,
)


import torch

from experiments.fitting.datasets.spherical.RbfInterpolator import RBFInterpolator
from scipy.stats import vonmises, truncnorm

SLACK = 0.0


class SphericalCoordinateDataset(CoordinateDataset):
    def __init__(
        self,
        base_dataset,
        n_coords,
        num_pairs,
        precomputed_dir,
        x_min=None,
        x_max=None,
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
            save_data (bool): If True, load precomputed data into memory.
            force_recompute (bool): If True, force recomputation of precomputed data.
        """

        x_min = np.array(
            x_min if x_min is not None else [0.0, 0 * np.pi / 180], dtype=np.float32
        )
        x_max = np.array(
            x_max if x_max is not None else [2 * np.pi, np.pi - 0 * np.pi / 180],
            dtype=np.float32,
        )

        # Coordinate bounds
        super().__init__(
            base_dataset=base_dataset,
            n_coords=n_coords,
            num_pairs=num_pairs,
            dim_signal=2,
            precomputed_dir=precomputed_dir,
            x_min=x_min,
            x_max=x_max,
            interpolation_mode="rbf",
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

        velocity = velocity.T

        # Generate coordinate chunks
        coord_chunks = self._generate_coords()
        n_theta, n_phi = velocity.shape
        # Compute bin edges
        theta_edges = np.linspace(self.x_min[0], self.x_max[0], n_theta + 1)
        phi_edges = np.linspace(self.x_min[1], self.x_max[1], n_phi + 1)

        # Compute bin centers
        theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
        phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

        # Creating the coordinate grid for the unit sphere.
        x = np.outer(np.sin(phi_centers), np.cos(theta_centers))
        y = np.outer(np.sin(phi_centers), np.sin(theta_centers))
        z = np.outer(np.cos(phi_centers), np.ones_like(phi_centers))

        points = np.stack([x[..., None], y[..., None], z[..., None]], axis=-1)
        points = points.reshape(-1, 3)

        x, y, z = np.array(points).T
        velocity = velocity.reshape(-1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        interpolator = RBFInterpolator(
            torch.tensor(points, dtype=torch.float32, device=device),
            torch.tensor(velocity, dtype=torch.float32, device=device),
            kernel="thin_plate_spline",
            spherical=True,
            device=device,
        )

        precomputed_chunks = []

        for chunk in coord_chunks:
            # Use NumPy arrays instead of immediately converting to JAX arrays
            # This is more efficient as JAX will convert them when needed
            theta = chunk[..., 0]
            phi = chunk[..., 1]

            X = np.sin(phi) * np.cos(theta)
            Y = np.sin(phi) * np.sin(theta)
            Z = np.cos(phi)

            p = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

            interpolated_values = (
                interpolator(torch.tensor(p, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
            )

            interpolated_values = interpolated_values.reshape(self.num_pairs, 2)

            coords_pairs = np.array(chunk, dtype=np.float32).reshape(
                self.num_pairs, 2, 2
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
        coords = np.random.rand(self.n_coords, 2)  # Random in [0, 1]
        x_min = np.array(self.x_min, dtype=np.float32)
        x_max = np.array(self.x_max, dtype=np.float32)
        if x_min[1] == 0:
            x_min[1] = SLACK * np.pi / 180  # Avoid singularity at poles
        if x_max[1] == np.pi:
            x_max[1] = np.pi - SLACK * np.pi / 180

        coords = (x_max - x_min) * coords + x_min
        np.random.shuffle(coords)

        return [
            coords[i : i + self.num_pairs * 2]
            for i in range(0, self.n_coords, self.num_pairs * 2)
        ]

    def _generate_coords1(self, p_uniform=0.8, von_mises_kappa=5.0, normal_std=0.3):
        """
        Generate random coordinate chunks for a single sample with advanced sampling.

        Args:
            p_uniform (float): Probability of sampling second point uniformly (vs. centered sampling)
            von_mises_kappa (float): Concentration parameter for von Mises distribution
            normal_std (float): Standard deviation for normal distribution (in radians)
        """
        coords = []

        # Check if theta domain is [0, 2pi] (full circle)
        theta_is_full_circle = np.isclose(self.x_max[0] - self.x_min[0], 2 * np.pi)

        # Total number of pairs we need
        total_pairs_needed = self.n_coords // 2

        for _ in range(total_pairs_needed):
            # Sample first point uniformly
            theta1 = np.random.uniform(self.x_min[0], self.x_max[0])
            phi1 = np.random.uniform(self.x_min[1], self.x_max[1])

            # Flip coin to decide sampling strategy for second point
            use_uniform = np.random.binomial(1, p_uniform)

            if use_uniform:
                # Sample second point uniformly
                theta2 = np.random.uniform(self.x_min[0], self.x_max[0])
                phi2 = np.random.uniform(self.x_min[1], self.x_max[1])
            else:
                # Sample second point centered at (theta1 + pi, phi1)

                # For theta coordinate
                if theta_is_full_circle:
                    # Use von Mises distribution for circular domain
                    # Center at theta1 + pi, wrapped to [0, 2pi]
                    center_theta = (theta1 + np.pi) % (2 * np.pi)

                    # Sample from von Mises centered at 0, then shift
                    vm_sample = vonmises.rvs(kappa=von_mises_kappa, loc=0)
                    theta2 = (center_theta + vm_sample) % (2 * np.pi)

                else:
                    # Use truncated normal for non-circular domain
                    center_theta = theta1 + np.pi

                    # Handle wrapping manually if center is outside domain
                    if center_theta > self.x_max[0]:
                        center_theta = center_theta - 2 * np.pi
                    elif center_theta < self.x_min[0]:
                        center_theta = center_theta + 2 * np.pi

                    # If center is still outside domain, just use uniform sampling
                    if center_theta < self.x_min[0] or center_theta > self.x_max[0]:
                        theta2 = np.random.uniform(self.x_min[0], self.x_max[0])
                    else:
                        # Use truncated normal
                        a = (self.x_min[0] - center_theta) / normal_std
                        b = (self.x_max[0] - center_theta) / normal_std
                        theta2 = truncnorm.rvs(a, b, loc=center_theta, scale=normal_std)

                # For phi coordinate (never wraps)
                # Use truncated normal centered at phi1
                a = (self.x_min[1] - phi1) / normal_std
                b = (self.x_max[1] - phi1) / normal_std
                phi2 = truncnorm.rvs(a, b, loc=phi1, scale=normal_std)

            # Add both points to coords
            coords.append([theta1, phi1])
            coords.append([theta2, phi2])

        coords = np.array(coords, dtype=np.float32)

        # Shuffle all coordinates
        np.random.shuffle(coords)

        # Split into chunks
        chunks = []
        for i in range(0, len(coords), self.num_pairs * 2):
            chunk = coords[i : i + self.num_pairs * 2]
            if len(chunk) == self.num_pairs * 2:  # Only add full chunks
                chunks.append(chunk)

        return chunks


def create_spherical_dataloader(
    base_dataset,
    batch_size,
    n_coords,
    num_pairs,
    precomputed_dir,
    shuffle=True,
    x_min=None,
    x_max=None,
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
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to use pinned memory.
        save_data (bool): If True, load precomputed data into memory.
        force_recompute (bool): If True, force recomputation of precomputed data.

    Returns:
        DataLoader: The DataLoader for the CoordinateDataset.
    """

    dataset = SphericalCoordinateDataset(
        base_dataset=base_dataset,
        n_coords=n_coords,
        num_pairs=num_pairs,
        x_min=x_min,
        x_max=x_max,
        precomputed_dir=precomputed_dir,
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
