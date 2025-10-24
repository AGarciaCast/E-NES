import numpy as np


from torch.utils.data import Dataset

import os
import json
import numpy as np
from tqdm import tqdm


import os
import glob
import json
import numpy as np
from torch.utils.data import Dataset
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt

import scipy as sp
import experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp as eikivp
from experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.vesselness import (
    VesselnessSE2,
)
from experiments.fitting.utils.ground_truth.IterativeEikonal.eikivp.SE2.costfunction import (
    CostSE2,
)
import taichi as ti

ti.init(arch=ti.cpu, debug=False)

LAMBDA = 500
POWER_COST = 5


class DrivePatchCachedDataset(Dataset):
    def __init__(
        self,
        train=True,
        patch_dim=32,
        total_patches=100,
        vmin=0.1,
        lift_dim=0,
        force_recompute=False,
        σ_s_list=np.array((1, 2)),
        σ_o=0.5 * 0.75**2,
        σ_s_ext=1,
        σ_o_ext=0.01,
    ):
        """
        A dataset that extracts random patches from DRIVE images and caches them.
        Only patches with nonempty segmentation masks are kept.

        Args:
            root_dir (str): Path to the DRIVE dataset root (e.g. '../../Data/DRIVE/training').
            patch_dim (int): Dimension of the square patch.
            total_patches (int): Total number of patches to generate.
            precomputed_dir (str or None): Directory to store cached patches. If None, caching is disabled.
            transform (callable, optional): Optional transform to apply on the image patch.
            save_data (bool): If True, load cached data into memory.
            force_recompute (bool): If True, force recomputation of patches even if cache exists.
        """
        self.patch_dim = patch_dim
        self.total_patches = total_patches
        self.vmin = vmin
        self.lift_dim = lift_dim
        self.train = train
        self.force_recompute = force_recompute
        self.σ_s_list = σ_s_list
        self.σ_o = σ_o
        self.σ_s_ext = σ_s_ext
        self.σ_o_ext = σ_o_ext

        if train:
            precomputed_dir = "./experiments/fitting/datasets/position_orientation/data/DRIVE/training/1st_manual"

            self.mask_paths = sorted(glob.glob(os.path.join(precomputed_dir, "*.gif")))
        else:
            precomputed_dir = "./experiments/fitting/datasets/position_orientation/data/STARE/labels-ah"
            self.mask_paths = sorted(glob.glob(os.path.join(precomputed_dir, "*.ppm")))

        self.patches_dir = os.path.join(precomputed_dir, "patches")
        self.velocity_dir = os.path.join(precomputed_dir, "velocity")

        if not self.mask_paths:
            raise ValueError("No images or masks found in the specified directory.")

        # Metadata file
        self.metadata_path = os.path.join(self.patches_dir, "metadata.json")
        os.makedirs(self.patches_dir, exist_ok=True)

        # Precompute or validate precomputed data
        self._prepare_data()

    def _prepare_data(self):
        """
        Ensures precomputed data is consistent with metadata. Recomputes if necessary.
        """
        metadata = self._load_metadata()

        # Check if recomputation is needed

        all_files_exist = self._all_files_exist()
        ###################### START: MODIFY THIS PART ######################

        # ADD THE VESSELNESS HYPERPARAMETERS HERE
        current_params = {
            "patch_dim": self.patch_dim,
            "total_patches": self.total_patches,
            "vmin": self.vmin,
            "lift_dim": self.lift_dim,
        }
        ###################### END: MODIFY THIS PART ######################

        need_recompute = (
            self.force_recompute
            or metadata == None
            or metadata != current_params
            or not all_files_exist
        )

        if need_recompute:
            self._clear_precomputed_data()
            self._compute_all()
            self._save_metadata(current_params)

    def _all_files_exist(self):
        """Check if all expected precomputed files exist. Moreover, find vmin and vmax of entire dataset."""
        aux = []
        for idx in range(self.total_patches):
            aux.append(
                os.path.exists(os.path.join(self.patches_dir, f"patch_{idx}.npy"))
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
        for file in os.scandir(self.patches_dir):
            if file.is_file():
                os.remove(file.path)

    def _compute_all(self):
        """Compute and save precomputed data for all samples."""
        self._compute_velocity_all()
        for idx in tqdm(range(self.total_patches), desc="Precomputing Vessel Dataset"):
            self._compute_sample(idx)

    def _compute_velocity_all(self):
        for idx in tqdm(
            range(len(self.mask_paths)), desc="Precomputing Velocities Dataset"
        ):
            self._compute_velocity(idx)

    def _compute_velocity(self, idx):
        ti.init(
            debug=False,
            log_level=ti.ERROR,
        )
        # import image
        mask = io.imread(self.mask_paths[idx])
        im_name = os.path.splitext(os.path.basename(self.mask_paths[idx]))[0]
        if self.train:
            mask = mask[0]
        mask = mask.T

        # rescale patch to values in [0,1] and reverse colors for computing vesselness
        mask = mask / 255.0
        mask = 1 - mask
        # remove high frequency
        u = mask - sp.ndimage.gaussian_filter(mask, 16, truncate=2.0, mode="nearest")

        # set parameters for lifting
        dim_I, dim_J = mask.shape
        dim_K = (self.lift_dim != 0) * self.lift_dim + (
            self.lift_dim == 0
        ) * 32  # 32 if lift_dim == 0, otherwise lift_dim
        Is, Js, Ks = np.indices((dim_I, dim_J, dim_K))
        dxy = 1.0
        x_min, x_max = 0.0, dim_I - 1.0
        y_min, y_max = 0.0, dim_J - 1.0
        θ_min, θ_max = 0.0, 2.0 * np.pi
        dxy = (x_max - x_min) / (dim_I - 1)
        dθ = (θ_max - θ_min) / dim_K
        xs, ys, θs = eikivp.SE2.utils.coordinate_array_to_real(
            Is, Js, Ks, x_min, y_min, θ_min, dxy, dθ
        )

        # create cakewavelets
        cws = eikivp.orientationscore.cakewavelet_stack(
            min(dim_I, dim_J), dim_K, inflection_point=0.8
        )
        # calculate orientation score U (in SE(2))
        U = np.transpose(
            eikivp.orientationscore.wavelet_transform(u, cws), axes=(1, 2, 0)
        ).real

        # create mask (no boundary effects in images, so only ones)
        mask = np.ones_like(U)

        # calculate vesselness
        V = VesselnessSE2(
            self.σ_s_list,
            self.σ_o,
            self.σ_s_ext,
            self.σ_o_ext,
            os.path.basename(self.mask_paths[idx]),
        )
        V.compute_V(U, mask, θs, dxy, dθ, bifurcations=None)

        # calculate cost function
        C = CostSE2(V, LAMBDA, POWER_COST)

        # calculate velocity field
        velocity = 1 / C.C
        if self.lift_dim == 0:
            velocity = np.max(velocity, axis=-1)

        mask = velocity

        np.save(os.path.join(self.velocity_dir, f"{im_name}.npy"), mask)

    def _compute_sample(self, idx):

        half = self.patch_dim // 2
        patch_mask = None

        while True:
            img_idx = np.random.randint(len(self.mask_paths))
            mask = io.imread(self.mask_paths[img_idx])
            im_name = os.path.splitext(os.path.basename(self.mask_paths[img_idx]))[0]
            velocity = np.load(os.path.join(self.velocity_dir, f"{im_name}.npy"))

            if self.train:
                mask = mask[0]
            mask = mask.T

            h, w = mask.shape

            if h <= self.patch_dim or w <= self.patch_dim:
                raise ValueError("Image dimensions are smaller than the patch size.")
            # Randomly choose a center point
            i = np.random.randint(half, h - half)
            j = np.random.randint(half, w - half)
            patch_mask = mask[i - half : i + half + 1, j - half : j + half + 1]
            patch_velocity = velocity[i - half : i + half + 1, j - half : j + half + 1]
            # Only accept if mask patch is not empty (has at least one vessel pixel)
            if np.sum(patch_mask > 0) > int(self.patch_dim * self.patch_dim * 0.2):
                break

        np.save(os.path.join(self.patches_dir, f"patch_{idx}.npy"), patch_mask)
        # patch_velocity = (patch_velocity - 1) / 500 + 1
        np.save(
            os.path.join(self.patches_dir, f"patch_{idx}_velocity.npy"), patch_velocity
        )

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        """Retrieve a specific chunk."""
        precomputed_file = os.path.join(self.patches_dir, f"patch_{idx}_velocity.npy")
        data = np.load(precomputed_file)

        return data, np.array([])


if __name__ == "__main__":
    dataset = DrivePatchCachedDataset(
        train=True, total_patches=100, patch_dim=65, force_recompute=False, lift_dim=32
    )

    # plot dataset[0][0] where we take the min of the 3 dimension. we plot with igmshow
    fig, ax = plt.subplots(figsize=[5, 5])
    ax.imshow(np.max(dataset[0][0], axis=2).T, cmap=plt.cm.bone)
    fig.savefig("patch.png")

    print(len(dataset), dataset[0][0].shape)
