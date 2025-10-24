import numpy as np
from torch.utils.data import Dataset

import jax.numpy as jnp
from jax import jit
import jax


def spherical_to_cartesian(inputs):
    """
    Convert spherical coordinates to Cartesian coordinates on unit sphere.

    Args:
        inputs: Array of shape (..., 2) where inputs[..., 0] is theta and inputs[..., 1] is phi
                theta ∈ [0, 2π]: azimuthal angle
                phi ∈ [0, π]: polar angle (from z-axis)

    Returns:
        Array of shape (..., 3) with Cartesian coordinates (x, y, z)
    """
    theta = inputs[..., 0]
    phi = inputs[..., 1]

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    cos_p = jnp.cos(phi)
    sin_p = jnp.sin(phi)

    transformed_inputs = jnp.stack(
        [
            cos_t * sin_p,  # x
            sin_t * sin_p,  # y
            cos_p,  # z
        ],
        axis=-1,
    )

    return transformed_inputs


@jit
def von_mises_fisher_density(spherical_coords, mu, kappa):
    """
    Evaluate the von Mises-Fisher density on the unit sphere.

    Args:
        spherical_coords: Array of shape (..., 2) with (theta, phi) coordinates
        mu: Mean direction vector of shape (3,) - must be unit vector
        kappa: Concentration parameter (scalar) - kappa >= 0

    Returns:
        Density values of shape (...)
    """
    # Convert spherical to Cartesian coordinates
    x = spherical_to_cartesian(spherical_coords)  # shape (..., 3)

    # Ensure mu is unit vector
    mu = mu / jnp.linalg.norm(mu)

    # Handle the case where kappa is very small (uniform distribution)
    uniform_density = 1.0 / (4.0 * jnp.pi)

    def concentrated_case():
        # Normalization constant: kappa / (4π * sinh(kappa))
        normalization = kappa / (4.0 * jnp.pi * jnp.sinh(kappa))

        # Dot product between x and mu
        dot_product = jnp.sum(x * mu, axis=-1)  # shape (...)

        # Density: normalization * exp(kappa * mu^T * x)
        density = normalization * jnp.exp(kappa * dot_product)
        return density

    def uniform_case():
        # Return uniform density for all points
        return jnp.full(x.shape[:-1], uniform_density)

    # Use conditional to handle kappa ≈ 0 case
    return jax.lax.cond(kappa < 1e-10, uniform_case, concentrated_case)


class RandomSphericalGaussian(Dataset):
    def __init__(
        self,
        num_samples,
        size=10,
        vmin=1,
        vmax=10,
        kappa_min=1,
        kappa_max=5.0,
        transform=None,
        x_min=None,
        x_max=None,
    ):
        """
        Generate random von Mises-Fisher distributions on the sphere.

        Args:
            num_samples: Number of samples to generate
            size: Grid size (size x size pixels on the sphere)
            vmin: Minimum value for the generated functions
            vmax: Maximum value for the generated functions
            kappa_min: Minimum concentration parameter
            kappa_max: Maximum concentration parameter
            transform: Optional transform to apply to the generated data
        """
        data = []
        coords = []

        # Create spherical coordinate grid
        n_theta, n_phi = size, size

        x_min = np.array(
            x_min if x_min is not None else [0.0, 0 * np.pi / 180], dtype=np.float32
        )
        x_max = np.array(
            x_max if x_max is not None else [2 * np.pi, np.pi - 0 * np.pi / 180],
            dtype=np.float32,
        )
        theta_vals = np.linspace(x_min[0], x_max[0], n_theta, endpoint=False)
        phi_vals = np.linspace(x_min[1], x_max[1], n_phi, endpoint=False)

        # Add half-pixel offset to get pixel centers
        theta_step = 2 * np.pi / n_theta
        phi_step = np.pi / n_phi
        theta_vals += theta_step / 2
        phi_vals += phi_step / 2

        # Create meshgrid of spherical coordinates
        Theta, Phi = np.meshgrid(theta_vals, phi_vals, indexing="ij")
        spherical_coords = np.stack([Theta, Phi], axis=-1)  # Shape: (size, size, 2)

        coords.append(spherical_coords)

        for _ in range(num_samples):
            # Generate random mean direction on the sphere
            # Sample uniformly on unit sphere
            mu_unnormalized = 2.0 * np.random.rand(3) - 1.0
            while np.linalg.norm(mu_unnormalized) > 1.0:
                mu_unnormalized = 2.0 * np.random.rand(3) - 1.0
            mu = mu_unnormalized / np.linalg.norm(mu_unnormalized)

            # Generate random concentration parameter
            kappa = kappa_min + (kappa_max - kappa_min) * np.random.rand()

            # Convert to JAX arrays
            mu_jax = jnp.array(mu)
            kappa_jax = jnp.array(kappa)
            spherical_coords_jax = jnp.array(spherical_coords)

            # Evaluate von Mises-Fisher density
            density = von_mises_fisher_density(spherical_coords_jax, mu_jax, kappa_jax)

            # Convert to numpy and scale to [vmin, vmax]
            density_np = np.array(density)

            # Normalize density to [0, 1] range first
            density_min = np.min(density_np)
            density_max = np.max(density_np)
            if density_max > density_min:
                density_normalized = (density_np - density_min) / (
                    density_max - density_min
                )
            else:
                density_normalized = np.ones_like(density_np) * 0.5

            # Scale to [vmin, vmax]
            V = vmin + (vmax - vmin) * (1 - density_normalized)

            if transform is not None:
                V = transform(V)
            data.append(V)

        self.data = data
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], np.array([])


# Example usage
if __name__ == "__main__":
    # Create dataset with spherical Gaussians
    dataset = RandomSphericalGaussian(
        num_samples=5, size=20, vmin=0, vmax=10, kappa_min=10, kappa_max=15.0
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Grid coordinates shape: {dataset.get_coordinates().shape}")

    # Get first sample
    sample, _ = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample value range: [{np.min(sample):.3f}, {np.max(sample):.3f}]")
