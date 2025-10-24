from torch.utils.data import Dataset

import os
import json
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf


class GroundTruthDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        solver,
        name,
        grid_data_fn,
        cfg,
        precomputed_dir,
        save_data=True,
        force_recompute=False,
    ):
        """
        CoordinateDataset generates coordinate pairs and their interpolated values
        for a given base dataset. Optionally caches precomputed data for faster access.

        Args:
            base_dataset (Dataset): The base dataset of velocity fields or similar.
            gt_solver.
            cfg.
            precomputed_dir.
            save_data (bool): If True, load precomputed data into memory.
            force_recompute (bool): If True, force recomputation of precomputed data.
        """
        self.base_dataset = base_dataset
        self.name = name
        self.gt_solver = solver
        self.grid_data = grid_data_fn(*base_dataset[0][0].shape)
        self.precomputed_dir = precomputed_dir
        self.save_data = save_data
        self.force_recompute = force_recompute
        self.cfg = cfg

        # Metadata file
        self.metadata_path = os.path.join(
            precomputed_dir, f"gt_metadata_{self.name}.json"
        )
        os.makedirs(precomputed_dir, exist_ok=True)

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

        current_params = OmegaConf.to_container(self.cfg, resolve=True)
        current_params = dict(current_params)
        del current_params["num_visualized"]
        del current_params["active"]
        del current_params["save_data"]
        del current_params["force_recompute"]

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
            aux.append(
                os.path.exists(
                    os.path.join(self.precomputed_dir, f"gt_{self.name}_{idx}.npy")
                )
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
            if file.is_file() and f"gt_{self.name}_" in file.path:
                os.remove(file.path)

    def _compute_all(self):
        """Compute and save precomputed data for all samples."""
        for idx in tqdm(range(self.total_samples), desc="Precomputing Ground Truth"):
            self._compute_sample(idx)

    def _compute_sample(self, idx):
        velocity, _ = self.base_dataset[idx]
        T_ref, fmmTime, self.grid_data = self.gt_solver(velocity, self.grid_data)

        result = {
            "T_ref": T_ref,
            "fmmTime": fmmTime,
        }

        # Save to disk
        np.save(os.path.join(self.precomputed_dir, f"gt_{self.name}_{idx}.npy"), result)

    def _load_all_to_memory(self):
        """Load all precomputed data into memory."""
        self.data = []
        for idx in range(self.total_samples):
            precomputed_file = os.path.join(
                self.precomputed_dir, f"gt_{self.name}_{idx}.npy"
            )
            self.data.append(np.load(precomputed_file, allow_pickle=True).item())

    def __len__(self):
        """Return the total number of chunks across all samples."""
        return self.total_samples

    def __getitem__(self, idx):
        """Retrieve a specific chunk."""

        if self.data is None:  # Load from disk if not cached
            precomputed_file = os.path.join(
                self.precomputed_dir, f"gt_{self.name}_{idx}.npy"
            )
            data = np.load(precomputed_file, allow_pickle=True).item()
        else:  # Use cached data
            data = self.data[idx]

        return data["T_ref"], data["fmmTime"]
