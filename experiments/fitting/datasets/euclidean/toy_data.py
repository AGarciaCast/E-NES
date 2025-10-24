import numpy as np
from torch.utils.data import Dataset


class HomogenousData(Dataset):
    def __init__(
        self, dimension, num_samples, size=10, vmin=1, vmax=10, transform=None
    ):
        data = []
        for _ in range(num_samples):
            # Generate a random constant in the range [vmin, vmax]
            rand_value = (vmax - vmin) * np.random.rand(1) + vmin

            # Create a tensor of ones with the given dimension and size
            ones_tensor = np.ones([size] * dimension)

            # Multiply the random constant by the ones tensor
            vel = rand_value * ones_tensor
            if transform is not None:
                vel = transform(vel)

            data.append(vel)

        self.data = data
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], np.array([])


class RandomGaussian(Dataset):
    def __init__(
        self, dimension, num_samples, size=10, vmin=1, vmax=10, transform=None
    ):
        data = []

        nx, ny = size, size
        xmin, xmax = -1.0, 1.0
        ymin, ymax = -1.0, 1.0

        x = np.linspace(xmin, xmax, nx)
        z = np.linspace(ymin, ymax, ny)

        Xr = np.stack(np.meshgrid(x, z, indexing="ij"), axis=-1)

        for _ in range(num_samples):
            mus = 2.0 * np.random.rand(dimension) - 1
            sigmas = 0.9 * np.random.rand(dimension) + 0.1
            V = (vmax - vmin) * (
                1 - np.exp(-((Xr - mus) ** 2 / 2 / sigmas**2).sum(axis=-1))
            ) + vmin
            if transform is not None:
                V = transform(V)
            data.append(V)

        self.data = data
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], np.array([])
