import os
import json
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset


class CoordinateDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        n_coords,
        num_pairs,
        dim_signal,
        precomputed_dir,
        x_min=None,
        x_max=None,
        interpolation_mode=None,
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
            interpolation_mode (str): Interpolation mode for grid sampling .
            save_data (bool): If True, load precomputed data into memory.
            force_recompute (bool): If True, force recomputation of precomputed data.
        """
        self.base_dataset = base_dataset
        self.n_coords = n_coords
        self.num_pairs = num_pairs
        self.dim_signal = dim_signal
        self.interpolation_mode = interpolation_mode
        self.precomputed_dir = precomputed_dir
        self.save_data = save_data
        self.force_recompute = force_recompute
        self.vmax = float("-inf")
        self.vmin = float("inf")

        # Coordinate bounds
        self.x_min = x_min
        self.x_max = x_max

        assert len(self.x_min) == self.dim_signal and len(self.x_max) == self.dim_signal

        # Metadata file
        self.metadata_path = os.path.join(precomputed_dir, "metadata.json")
        os.makedirs(precomputed_dir, exist_ok=True)

        # Number of chunks per sample
        self.chunks_per_sample = self.n_coords // (self.num_pairs * 2)
        self.total_samples = len(base_dataset)

        # Precompute or validate precomputed data
        self.data = None
        self._prepare_data()

    def _prepare_data(self):
        """
        Ensures precomputed data is consistent with metadata. Recomputes if necessary.
        """
        metadata = self._load_metadata()

        # Check if recomputation is needed

        all_files_exist = self._all_files_exist_min_max()

        current_params = {
            "n_coords": self.n_coords,
            "num_pairs": self.num_pairs,
            "dim_signal": self.dim_signal,
            "x_min": self.x_min.tolist() if self.x_min is not None else None,
            "x_max": self.x_max.tolist() if self.x_min is not None else None,
            "vmin": np.float64(self.vmin),
            "vmax": np.float64(self.vmax),
            "interpolation_mode": self.interpolation_mode,
        }

        need_recompute = (
            self.force_recompute
            or metadata == None
            or metadata != current_params
            or not all_files_exist
        )

        if need_recompute:
            self._clear_precomputed_data()
            self._save_metadata(current_params)
            self._compute_all()

        # Load precomputed data into memory if save_data is True
        if self.save_data:
            self._load_all_to_memory()

    def _all_files_exist_min_max(self):
        """Check if all expected precomputed files exist. Moreover, find vmin and vmax of entire dataset."""
        aux = []
        for idx in range(self.total_samples):
            velocity, _ = self.base_dataset[idx]
            self.vmin = min(np.min(velocity), self.vmin)
            self.vmax = max(np.max(velocity), self.vmax)
            aux.append(
                os.path.exists(os.path.join(self.precomputed_dir, f"sample_{idx}.npy"))
            )

        return all(aux)

    def _load_metadata(self):
        """Load metadata from disk, if it exists."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return None

    def _save_metadata(self, metadata):
        """Save metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

    def _clear_precomputed_data(self):
        """Remove all precomputed data files."""
        for file in os.scandir(self.precomputed_dir):
            if file.is_file():
                os.remove(file.path)

    def _compute_all(self):
        """Compute and save precomputed data for all samples."""
        for idx in tqdm(range(self.total_samples), desc="Precomputing data"):
            self._compute_sample(idx)

    def _compute_sample(self, idx):
        raise NotImplementedError("_compute_sample method must be implemented.")

    def _generate_coords(self):
        raise NotImplementedError("_generate_coords method must be implemented.")

    def _load_all_to_memory(self):
        """Load all precomputed data into memory."""
        self.data = []
        for idx in range(self.total_samples):
            precomputed_file = os.path.join(self.precomputed_dir, f"sample_{idx}.npy")
            self.data.append(np.load(precomputed_file, allow_pickle=True))

    def __len__(self):
        """Return the total number of chunks across all samples."""
        return self.total_samples * self.chunks_per_sample

    def __getitem__(self, idx):
        """Retrieve a specific chunk."""
        sample_idx = idx // self.chunks_per_sample
        chunk_idx = idx % self.chunks_per_sample

        if self.data is None:  # Load from disk if not cached
            precomputed_file = os.path.join(
                self.precomputed_dir, f"sample_{sample_idx}.npy"
            )
            precomputed_chunks = np.load(precomputed_file, allow_pickle=True)
        else:  # Use cached data
            precomputed_chunks = self.data[sample_idx]

        chunk_data = precomputed_chunks[chunk_idx]
        return chunk_data["interpolated_values"], chunk_data["coords_pairs"], sample_idx


def collate_batch_with_index(batch):
    """
    Custom collate function to handle batches of interpolated values and coordinate pairs.
    Uses NumPy arrays which will be converted to JAX arrays when passed to JAX functions.
    """
    interpolated_values = np.stack(
        [item[0] for item in batch]
    )  # (batch_size, num_pairs, 2, dim_channels)
    coords = np.stack(
        [item[1] for item in batch]
    )  # (batch_size, num_pairs, 2, dim_signal)
    indices = np.array([item[2] for item in batch])  # (batch_size,)

    return interpolated_values, coords, indices


class create_index_dataset(Dataset):
    def __init__(self, data):
        # Creating identical pairs
        self.data = data

    def __getitem__(self, index):
        x, _ = self.data[index]
        return x, index

    def __len__(self):
        return len(self.data)
