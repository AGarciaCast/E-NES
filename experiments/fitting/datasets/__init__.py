from experiments.fitting.datasets.euclidean.euclidean_coordinate_dataset import (
    create_euclidean_dataloader,
)
from experiments.fitting.datasets.euclidean import get_base_euclidean_datasets
from experiments.fitting.datasets.position_orientation import (
    get_base_position_orientation_datasets,
)

from experiments.fitting.datasets.spherical import (
    get_base_spherical_datasets,
)


from experiments.fitting.datasets.position_orientation.position_orientation_coordinate_dataset import (
    create_position_orientation_dataloader,
)


from experiments.fitting.datasets.spherical.spherical_coordinate_dataset import (
    create_spherical_dataloader,
)


import math


def get_dataloaders(cfg, meta=False):

    space = cfg.geometry.input_space
    data_cfg = cfg.data
    seed = cfg.seed

    if meta:
        aux_txt = "meta_"
    else:
        aux_txt = ""

    if space == "Euclidean":
        train_dset, val_dset, test_dset = get_base_euclidean_datasets(
            data_cfg.base_dataset, cfg.geometry.dim_signal, seed
        )

        try:
            data_cfg.base_dataset.name = (
                cfg.data.base_dataset.name + "-3d"
                if data_cfg.base_dataset.create_3d
                else cfg.data.base_dataset.name
            )
        except:
            print(" ")

        train_loader = create_euclidean_dataloader(
            base_dataset=train_dset,
            batch_size=data_cfg.train_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            dim_signal=cfg.geometry.dim_signal,
            precomputed_dir=f"./experiments/fitting/datasets/euclidean/data/coord_{data_cfg.base_dataset.name}_{aux_txt}train",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.num_workers > 0,
            save_data=data_cfg.train_save_data,
            force_recompute=data_cfg.train_force_recompute,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )

        val_loader = create_euclidean_dataloader(
            base_dataset=val_dset,
            batch_size=data_cfg.test_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            dim_signal=cfg.geometry.dim_signal,
            precomputed_dir=f"./experiments/fitting/datasets/euclidean/data/coord_{data_cfg.base_dataset.name}_{aux_txt}val",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
            save_data=data_cfg.val_save_data,
            force_recompute=data_cfg.val_force_recompute,
            shuffle=False,
            seed=seed + 1,
        )

        test_loader = create_euclidean_dataloader(
            base_dataset=test_dset,
            batch_size=data_cfg.test_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            dim_signal=cfg.geometry.dim_signal,
            precomputed_dir=f"./experiments/fitting/datasets/euclidean/data/coord_{data_cfg.base_dataset.name}_{aux_txt}test",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
            save_data=data_cfg.test_save_data,
            force_recompute=data_cfg.test_force_recompute,
            shuffle=False,
            seed=seed + 2,
        )

        return train_loader, val_loader, test_loader

    elif space == "Position_Orientation":
        train_dset, val_dset, test_dset = get_base_position_orientation_datasets(
            data_cfg.base_dataset, seed
        )
        theta_range = (
            [
                0.0,
                2.0 * math.pi,
            ]
            if cfg.geometry.theta_range == "zero"
            else [-math.pi, math.pi]
        )

        train_loader = create_position_orientation_dataloader(
            base_dataset=train_dset,
            batch_size=data_cfg.train_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            precomputed_dir=f"./experiments/fitting/datasets/position_orientation/data/coord_{data_cfg.base_dataset.name}_xi_{cfg.geometry.metric.xi}_epsilon_{cfg.geometry.metric.epsilon}_{aux_txt}train",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            theta_range=theta_range,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.num_workers > 0,
            save_data=data_cfg.train_save_data,
            force_recompute=data_cfg.train_force_recompute,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )

        val_loader = create_position_orientation_dataloader(
            base_dataset=val_dset,
            batch_size=data_cfg.test_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            precomputed_dir=f"./experiments/fitting/datasets/position_orientation/data/coord_{data_cfg.base_dataset.name}_xi_{cfg.geometry.metric.xi}_epsilon_{cfg.geometry.metric.epsilon}_{aux_txt}val",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            theta_range=theta_range,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
            save_data=data_cfg.test_save_data,
            force_recompute=data_cfg.val_force_recompute,
            shuffle=False,
            seed=seed + 1,
        )

        test_loader = create_position_orientation_dataloader(
            base_dataset=test_dset,
            batch_size=data_cfg.test_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            precomputed_dir=f"./experiments/fitting/datasets/position_orientation/data/coord_{data_cfg.base_dataset.name}_xi_{cfg.geometry.metric.xi}_epsilon_{cfg.geometry.metric.epsilon}_{aux_txt}test",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            theta_range=theta_range,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
            save_data=data_cfg.test_save_data,
            force_recompute=data_cfg.test_force_recompute,
            shuffle=False,
            seed=seed + 2,
        )

        return train_loader, val_loader, test_loader

    elif space == "Spherical":
        assert (
            cfg.geometry.dim_signal == 3
        ), "Spherical space requires dim_signal to be 3."

        train_dset, val_dset, test_dset = get_base_spherical_datasets(
            data_cfg.base_dataset,
            dim_signal=2,
            seed=seed,
            xmin=data_cfg.x_min,
            xmax=data_cfg.x_max,
        )

        train_loader = create_spherical_dataloader(
            base_dataset=train_dset,
            batch_size=data_cfg.train_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            precomputed_dir=f"./experiments/fitting/datasets/spherical/data/coord_{data_cfg.base_dataset.name}_{aux_txt}train",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=data_cfg.num_workers > 0,
            save_data=data_cfg.train_save_data,
            force_recompute=data_cfg.train_force_recompute,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )

        val_loader = create_spherical_dataloader(
            base_dataset=val_dset,
            batch_size=data_cfg.test_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            precomputed_dir=f"./experiments/fitting/datasets/spherical/data/coord_{data_cfg.base_dataset.name}_{aux_txt}val",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
            save_data=data_cfg.test_save_data,
            force_recompute=data_cfg.val_force_recompute,
            shuffle=False,
            seed=seed + 1,
        )

        test_loader = create_spherical_dataloader(
            base_dataset=test_dset,
            batch_size=data_cfg.test_batch_size,
            n_coords=data_cfg.n_coords,
            num_pairs=data_cfg.num_pairs,
            precomputed_dir=f"./experiments/fitting/datasets/spherical/data/coord_{data_cfg.base_dataset.name}_{aux_txt}test",
            x_min=data_cfg.x_min,
            x_max=data_cfg.x_max,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            persistent_workers=False,
            save_data=data_cfg.test_save_data,
            force_recompute=data_cfg.test_force_recompute,
            shuffle=False,
            seed=seed + 2,
        )

        return train_loader, val_loader, test_loader

    else:
        raise ValueError(f"Unknown space name: {space}")
