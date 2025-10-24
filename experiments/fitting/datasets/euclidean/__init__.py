from experiments.fitting.datasets.euclidean.open_fwi import OpenFWIDataset
from experiments.fitting.datasets.euclidean.toy_data import (
    HomogenousData,
    RandomGaussian,
)

# from experiments.fitting.datasets.euclidean.permeability_sim import PermeabilityDataset

import numpy as np
import torchvision.transforms as T

import torch
from torch.utils import data
import json
import sys

from pathlib import Path

from experiments.fitting.datasets.base_coordinate_dataset import create_index_dataset

DATASET_PATH = Path("./experiments/fitting/datasets/euclidean")


def get_base_euclidean_datasets(dataset_cfg, dim_signal, seed=0):
    if "fwi" in dataset_cfg.name:
        with open(DATASET_PATH / "utils_open_fwi/fwi_dataset_config.json") as f:
            dataset_name = dataset_cfg.name[4:].replace("_", "-")
            try:
                ctx = json.load(f)[dataset_name]
            except KeyError:
                print("Unsupported dataset.")
                sys.exit()

        custom_transform = T.Compose(
            [
                T.Lambda(lambda x: np.squeeze(x, axis=0)),
                T.Lambda(lambda x: np.flip(x, axis=0)),
                T.Lambda(lambda x: np.moveaxis(x, 0, -1)),
                T.Lambda(lambda x: x / 1000.0),
            ]
        )
        dataset_name = dataset_cfg.name[4:].replace("-", "_")
        train_dset = OpenFWIDataset(
            DATASET_PATH / f"utils_open_fwi/{dataset_name}_train.txt",
            preload=dataset_cfg.preload,
            transform=custom_transform,
            file_size=ctx["file_size"],
            create_3d=dataset_cfg.create_3d,
        )

        assert dataset_cfg.num_signals_train + dataset_cfg.num_signals_val <= len(
            train_dset
        )

        test_dset = OpenFWIDataset(
            DATASET_PATH / f"utils_open_fwi/{dataset_name}_val.txt",
            preload=dataset_cfg.preload,
            transform=custom_transform,
            file_size=ctx["file_size"],
            create_3d=dataset_cfg.create_3d,
        )

        assert dataset_cfg.num_signals_test <= len(test_dset)

    elif dataset_cfg.name == "homogenous":

        train_dset = HomogenousData(
            dimension=dim_signal,
            num_samples=dataset_cfg.num_signals_train + dataset_cfg.num_signals_val,
            size=dataset_cfg.size,
            vmin=dataset_cfg.vmin_train,
            vmax=dataset_cfg.vmax_train,
        )

        test_dset = HomogenousData(
            dimension=dim_signal,
            num_samples=dataset_cfg.num_signals_test,
            size=dataset_cfg.size,
            vmin=dataset_cfg.vmin_test,
            vmax=dataset_cfg.vmax_test,
        )

    elif dataset_cfg.name == "gauss":

        train_dset = RandomGaussian(
            dimension=dim_signal,
            num_samples=dataset_cfg.num_signals_train + dataset_cfg.num_signals_val,
            size=dataset_cfg.size,
            vmin=dataset_cfg.vmin_train,
            vmax=dataset_cfg.vmax_train,
        )

        test_dset = RandomGaussian(
            dimension=dim_signal,
            num_samples=dataset_cfg.num_signals_test,
            size=dataset_cfg.size,
            vmin=dataset_cfg.vmin_test,
            vmax=dataset_cfg.vmax_test,
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
