import numpy as np
import openpnm as op
import porespy as ps

from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset

import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt  # Make sure matplotlib is imported


import numpy as np

import torch
from torch.utils import data
import json

from pathlib import Path

from experiments.fitting.datasets.base_coordinate_dataset import create_index_dataset


def generate_porous_medium(size=128, porosity=0.5):
    blobiness = np.random.uniform(1.0, 1.25)
    while True:
        im = ps.generators.blobs(
            shape=[size, size], porosity=porosity, blobiness=blobiness
        )

        inlets = np.zeros_like(im)
        inlets[0, :] = True
        outlets = np.zeros_like(im)
        outlets[-1, :] = True

        x = ps.filters.trim_nonpercolating_paths(im=im, inlets=inlets, outlets=outlets)

        if np.sum(x) > 0:
            return x.astype(np.float32)


def compute_permeability(im, voxel_size=1e-6, delta_P=1):
    size = im.shape[1]
    snow_output = ps.networks.snow2(im, voxel_size=voxel_size)
    pn = op.io.network_from_porespy(snow_output.network)

    # Assign labels
    pn["pore.left"] = pn["pore.xmin"]
    pn["pore.right"] = pn["pore.xmax"]
    pn["pore.diameter"] = pn["pore.inscribed_diameter"]
    pn["throat.diameter"] = pn["throat.inscribed_diameter"]
    pn["throat.coords"] = pn["throat.global_peak"]
    pn.add_model(
        propname="throat.length",
        model=op.models.geometry.throat_length.hybrid_cones_and_cylinders,
    )
    pn.add_model(
        propname="throat.hydraulic_size_factors",
        model=op.models.geometry.hydraulic_size_factors.cones_and_cylinders,
    )

    # Remove isolated pores that ps.filters.trim_nonpercolating_paths did not remove
    pn.add_model(propname="pore.cluster_number", model=op.models.network.cluster_number)
    pn.add_model(propname="pore.cluster_size", model=op.models.network.cluster_size)
    Ps = pn["pore.cluster_size"] < max(pn["pore.cluster_size"])
    op.topotools.trim(network=pn, pores=Ps)

    pn.regenerate_models()

    phase = op.phase.Water(network=pn)

    phase.add_model(
        propname="throat.hydraulic_conductance",
        model=op.models.physics.hydraulic_conductance.generic_hydraulic,
    )

    inlet = pn.pores("left")
    outlet = pn.pores("right")
    flow = op.algorithms.StokesFlow(network=pn, phase=phase)

    flow.set_value_BC(pores=inlet, values=delta_P)
    flow.set_value_BC(pores=outlet, values=0)
    flow.run()

    Q = np.abs(flow.rate(pores=outlet, mode="group")[0])
    A = np.pi * (pn["throat.diameter"].max() * 0.5) ** 2
    L = size * voxel_size
    mu = np.mean(phase["pore.viscosity"])
    K = Q * L * mu / (A * delta_P)

    KmD = K / 0.98e-12 * 1000

    return KmD


class PermeabilityDataset(Dataset):
    def __init__(
        self,
        size,
        num_samples,
        porosity=0.5,
        voxel_size=1e-6,
        delta_P=1,
        vmin=0.1,
        normalize=True,
        gaussian=False,
        train=True,
        iqr_multiplier=1.5,  # multiplier for IQR-based outlier removal
        target_std=None,  # if set, enforce the final permeability std to be below this value
    ):
        """
        Generates porous medium samples and computes their permeability. Optionally
        enforces a smaller standard deviation in the permeability distribution by
        regenerating samples that deviate far from the mean.
        """
        self.precomputed_dir = (
            "./experiments/fitting/datasets/euclidean/data/permeability"
        )

        if train:
            self.precomputed_dir = os.path.join(self.precomputed_dir, "train")
        else:
            self.precomputed_dir = os.path.join(self.precomputed_dir, "test")

        self.size = size
        self.num_samples = num_samples
        self.porosity = porosity
        self.voxel_size = voxel_size
        self.delta_P = delta_P
        self.normalize = normalize
        self.vmin = vmin
        self.iqr_multiplier = iqr_multiplier
        self.target_std = target_std
        self.train = train
        self.gaussian = gaussian

        # Metadata file
        self.metadata_path = os.path.join(self.precomputed_dir, "perm_metadata.json")
        os.makedirs(self.precomputed_dir, exist_ok=True)

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
            "size": self.size,
            "num_samples": self.num_samples,
            "porosity": self.porosity,
            "voxel_size": self.voxel_size,
            "kmin": float("+inf") if metadata is None else metadata["kmin"],
            "kmax": float("-inf") if metadata is None else metadata["kmax"],
            "mean_porosity": -1.0 if metadata is None else metadata["mean_porosity"],
            "std_porosity": -1.0 if metadata is None else metadata["std_porosity"],
            "mean_permeability": (
                -1.0 if metadata is None else metadata["mean_permeability"]
            ),
            "std_permeability": (
                -1.0 if metadata is None else metadata["std_permeability"]
            ),
            "delta_P": self.delta_P,
        }
        need_recompute = (
            metadata is None or metadata != current_params or not all_files_exist
        )

        if need_recompute:
            self._clear_precomputed_data()
            self._compute_all()
            current_params["kmin"] = self.kmin
            current_params["kmax"] = self.kmax
            current_params["mean_porosity"] = self.mean_porosity
            current_params["std_porosity"] = self.std_porosity
            current_params["mean_permeability"] = self.mean_permeability
            current_params["std_permeability"] = self.std_permeability
            self._save_metadata(current_params)
        else:
            self.kmin = metadata["kmin"]
            self.kmax = metadata["kmax"]
            self.mean_porosity = metadata["mean_porosity"]
            self.std_porosity = metadata["std_porosity"]
            self.mean_permeability = metadata["mean_permeability"]
            self.std_permeability = metadata["std_permeability"]

        print(f"Mean porosity: {self.mean_porosity:.2f}")
        print(f"Std porosity: {self.std_porosity:.2f}")
        print(f"Mean KmD: {self.mean_permeability:.2f}")
        print(f"Std KmD: {self.std_permeability:.2f}")

    def _all_files_exist_min_max(self):
        """Check if all expected precomputed files exist."""
        return all(
            os.path.exists(os.path.join(self.precomputed_dir, f"sample_{idx}.pt"))
            for idx in range(self.num_samples)
        )

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
        self.kmin = float("+inf")
        self.kmax = float("-inf")
        porosity_list = []
        KmD_list = []

        # First compute all samples
        for idx in tqdm(
            range(self.num_samples), desc="Precomputing Permeability Dataset"
        ):
            porosity, KmD = self._compute_sample(idx)
            porosity_list.append(porosity)
            KmD_list.append(KmD)

        # -------------------------------
        # Outlier removal using the IQR method
        # -------------------------------
        def get_outlier_indices(data, multiplier):
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            indices = [
                i for i, x in enumerate(data) if x < lower_bound or x > upper_bound
            ]
            return indices, lower_bound, upper_bound

        outlier_indices, lb, ub = get_outlier_indices(KmD_list, self.iqr_multiplier)
        iteration = 0
        while len(outlier_indices) > 0:
            print(
                f"Outlier removal iteration {iteration}: Found {len(outlier_indices)} outliers (KmD < {lb:.2f} or > {ub:.2f}). Regenerating these samples."
            )
            for idx in outlier_indices:
                porosity, KmD = self._compute_sample(idx)
                porosity_list[idx] = porosity  # update porosity value
                KmD_list[idx] = KmD  # update permeability value
            outlier_indices, lb, ub = get_outlier_indices(KmD_list, self.iqr_multiplier)
            iteration += 1
        print(
            "Outlier removal complete: No outliers remain based on the IQR criterion."
        )
        # -------------------------------

        # -------------------------------
        # Enforce a smaller standard deviation, if requested
        # -------------------------------
        if self.target_std is not None:
            max_std_iterations = 5000  # to prevent an infinite loop
            std_iter = 0
            while np.std(KmD_list) > self.target_std and std_iter < max_std_iterations:
                current_std = np.std(KmD_list)
                mean_perm = np.mean(KmD_list)
                # Identify the sample with the largest absolute deviation from the mean
                deviations = np.abs(np.array(KmD_list) - mean_perm)
                idx_to_replace = int(np.argmax(deviations))
                print(
                    f"Std {current_std:.2f} > target_std {self.target_std:.2f}. Regenerating sample {idx_to_replace} (deviation: {deviations[idx_to_replace]:.2f})."
                )
                porosity, KmD = self._compute_sample(idx_to_replace)
                porosity_list[idx_to_replace] = porosity
                KmD_list[idx_to_replace] = KmD
                std_iter += 1
            if std_iter >= max_std_iterations:
                print("Reached maximum iterations while enforcing target std.")
            else:
                print(
                    f"Final permeability std: {np.std(KmD_list):.2f} mD (target: {self.target_std:.2f} mD)."
                )

        iteration = 0
        while len(outlier_indices) > 0:
            print(
                f"Outlier removal iteration {iteration}: Found {len(outlier_indices)} outliers (KmD < {lb:.2f} or > {ub:.2f}). Regenerating these samples."
            )
            for idx in outlier_indices:
                porosity, KmD = self._compute_sample(idx)
                porosity_list[idx] = porosity  # update porosity value
                KmD_list[idx] = KmD  # update permeability value
            outlier_indices, lb, ub = get_outlier_indices(KmD_list, self.iqr_multiplier)
            iteration += 1
        print(
            "Outlier removal complete: No outliers remain based on the IQR criterion."
        )
        # -------------------------------

        self.mean_porosity = np.mean(porosity_list)
        self.std_porosity = np.std(porosity_list)
        self.mean_permeability = np.mean(KmD_list)
        self.std_permeability = np.std(KmD_list)
        self.kmin = min(KmD_list)
        self.kmax = max(KmD_list)

        # Plot boxplot of permeability values after processing
        fig, ax = plt.subplots()
        ax.boxplot(KmD_list)
        ax.set_title("Permeability values after outlier removal and std enforcement")
        fig.savefig(
            f"./experiments/fitting/figures/porous_medium_stats{'_train' if self.train else '_test'}.png",
            dpi=300,
        )

    def _compute_sample(self, idx):
        while True:
            try:
                porous_medium = generate_porous_medium(
                    size=self.size, porosity=self.porosity
                )
                # Ensure the network is valid for permeability computation
                _ = ps.networks.snow2(porous_medium, voxel_size=self.voxel_size)
                break
            except Exception as e:
                pass

        KmD = compute_permeability(
            porous_medium, voxel_size=self.voxel_size, delta_P=self.delta_P
        )
        self.kmin = min(self.kmin, KmD)
        self.kmax = max(self.kmax, KmD)
        result = {
            "porous_medium": porous_medium,
            "KmD": KmD,
        }

        # Save sample to disk
        torch.save(result, os.path.join(self.precomputed_dir, f"sample_{idx}.pt"))

        return ps.metrics.porosity(porous_medium), KmD

    def __len__(self):
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Retrieve a specific sample."""
        precomputed_file = os.path.join(self.precomputed_dir, f"sample_{idx}.pt")
        data = torch.load(precomputed_file, weights_only=False)

        porous_medium = data["porous_medium"]

        if self.gaussian:

            porous_medium = 0.2 + (1 - 0.2) * porous_medium
            porous_medium = gaussian_filter(porous_medium, sigma=3)

        if self.normalize:
            normKmD = (data["KmD"] - self.kmin) / (self.kmax - self.kmin)
            aux_max = np.max(porous_medium)
            aux_min = np.min(porous_medium)
            normPorousMedium = (porous_medium - aux_min) / (aux_max - aux_min)

            normPorousMedium = self.vmin + (1 - self.vmin) * normPorousMedium

            # normPorousMedium += data["porous_medium"]
            # normPorousMedium *= data["porous_medium"]
            # normPorousMedium = self.vmin + (1 - self.vmin) * normPorousMedium
            return normPorousMedium, normKmD
        else:
            return porous_medium, data["KmD"]


if __name__ == "__main__":
    np.random.seed(0)

    # You can adjust 'iqr_multiplier' and 'target_std' as needed.
    train_dset = PermeabilityDataset(
        size=128,
        num_samples=150,
        porosity=0.4,
        voxel_size=2.5e-6,
        delta_P=1,
        vmin=0.1,
        normalize=True,
        iqr_multiplier=1.5,  # using a stricter IQR criterion
        target_std=50,  # enforce that the std of KmD is below 5 mD (example value)
    )

    train_dset_1 = PermeabilityDataset(
        size=128,
        num_samples=150,
        porosity=0.4,
        voxel_size=2.5e-6,
        delta_P=1,
        vmin=0.1,
        normalize=True,
        gaussian=True,
        iqr_multiplier=1.5,  # using a stricter IQR criterion
        target_std=50,  # enforce that the std of KmD is below 5 mD (example value)
    )

    generator = torch.Generator().manual_seed(0)

    desired_splits = [100, 50]
    total_length = len(train_dset)
    if sum(desired_splits) != total_length:
        desired_splits.append(total_length - sum(desired_splits))

    splits = data.random_split(
        train_dset,
        desired_splits,
        generator,
    )
    train_dset = create_index_dataset(splits[0])
    val_dset = create_index_dataset(splits[1])

    total_length = len(train_dset)

    desired_splits = [
        50,
        50,
    ]
    train_dset, aug_dset = data.random_split(
        train_dset,
        desired_splits,
        generator,
    )
    aug_dset.dataset.data.dataset = train_dset_1

    print(train_dset.dataset.data.dataset.gaussian)
    train_dset = torch.utils.data.ConcatDataset([train_dset, aug_dset])

    train_dset = create_index_dataset(train_dset)

    # Plot 4 samples with the permeability value in the subtitle
    fig, axs = plt.subplots(2, 2, figsize=[10, 10])
    for i, ax in enumerate(axs.flat):
        sample_image, permeability = train_dset_1[i]
        ax.imshow(sample_image.T)
        ax.axis("off")
        ax.set_title(f"KmD: {permeability:.2f} mD")

    fig.savefig(
        "./experiments/fitting/figures/porous_medium_samples.png",
        dpi=300,
    )
