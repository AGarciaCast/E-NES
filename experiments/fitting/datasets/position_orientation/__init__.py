from experiments.fitting.datasets.euclidean.toy_data import (
    HomogenousData,
)

from experiments.fitting.datasets.position_orientation.maze import (
    MazeData,
)

from experiments.fitting.datasets.position_orientation.obstacles import (
    ObstaclesData,
)

from experiments.fitting.datasets.position_orientation.vessel import (
    DrivePatchCachedDataset,
)


import torch
from torch.utils import data


from pathlib import Path

from experiments.fitting.datasets.base_coordinate_dataset import create_index_dataset

DATASET_PATH = Path("./experiments/fitting/datasets/position_orientation")


def get_base_position_orientation_datasets(dataset_cfg, seed=0):

    if dataset_cfg.name == "homogenous":

        train_dset = HomogenousData(
            dimension=3,
            num_samples=dataset_cfg.num_signals_train + dataset_cfg.num_signals_val,
            size=dataset_cfg.size,
            vmin=dataset_cfg.vmin_train,
            vmax=dataset_cfg.vmax_train,
        )

        test_dset = HomogenousData(
            dimension=3,
            num_samples=dataset_cfg.num_signals_test,
            size=dataset_cfg.size,
            vmin=dataset_cfg.vmin_test,
            vmax=dataset_cfg.vmax_test,
        )

    elif "maze" in dataset_cfg.name:
        train_dset = MazeData(
            train=True,
            num_ori=dataset_cfg.lift_dim,
            vmin=dataset_cfg.vmin_train,
            vmax=dataset_cfg.vmax_train,
            wind=dataset_cfg.wind,
            gaussian=dataset_cfg.blur,
            seed=0,
        )

        test_dset = MazeData(
            train=False,
            num_ori=dataset_cfg.lift_dim,
            vmin=dataset_cfg.vmin_test,
            vmax=dataset_cfg.vmax_test,
            wind=dataset_cfg.wind,
            gaussian=dataset_cfg.blur,
            seed=1,
        )

    elif "obstacles" in dataset_cfg.name:
        train_dset = ObstaclesData(
            num_samples=dataset_cfg.num_signals_train + dataset_cfg.num_signals_val,
            size=dataset_cfg.size,
            num_circles=dataset_cfg.num_circles,
            radius=dataset_cfg.radius,
            num_ori=dataset_cfg.lift_dim,
            vmin=dataset_cfg.vmin_train,
            vmax=dataset_cfg.vmax_train,
            wind=dataset_cfg.wind,
            gaussian=dataset_cfg.blur,
            seed=0,
        )

        test_dset = ObstaclesData(
            num_samples=dataset_cfg.num_signals_test,
            size=dataset_cfg.size,
            num_circles=dataset_cfg.num_circles,
            radius=dataset_cfg.radius,
            num_ori=dataset_cfg.lift_dim,
            vmin=dataset_cfg.vmin_test,
            vmax=dataset_cfg.vmax_test,
            wind=dataset_cfg.wind,
            gaussian=dataset_cfg.blur,
            seed=1,
        )

    elif dataset_cfg.name == "vessel":

        train_dset = DrivePatchCachedDataset(
            train=True,
            patch_dim=dataset_cfg.patch_dim,
            total_patches=dataset_cfg.num_signals_train + dataset_cfg.num_signals_val,
            lift_dim=dataset_cfg.lift_dim,
        )

        test_dset = DrivePatchCachedDataset(
            train=False,
            patch_dim=dataset_cfg.patch_dim,
            total_patches=dataset_cfg.num_signals_test,
            lift_dim=dataset_cfg.lift_dim,
        )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_cfg.name}")

    generator = torch.Generator().manual_seed(seed)

    desired_splits = [dataset_cfg.num_signals_train, dataset_cfg.num_signals_val]
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

    total_length = len(test_dset)

    if dataset_cfg.num_signals_test < total_length:
        desired_splits = [
            dataset_cfg.num_signals_test,
            total_length - dataset_cfg.num_signals_test,
        ]
        test_dset, _ = data.random_split(
            test_dset,
            desired_splits,
            generator,
        )

    test_dset = create_index_dataset(test_dset)

    return train_dset, val_dset, test_dset
