import omegaconf
import argparse

from hydra import initialize, compose

from experiments.fitting.utils.logging import logger

import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")


from experiments.fitting.datasets import get_dataloaders
from experiments.fitting.trainers.ad_eikonal_trainer import AutoDecodingEikonalTrainer

from experiments.fitting.utils.visualization import (
    get_recon_visualization,
    get_gt_solver,
    get_gt_visualization,
    get_equiv_visualization,
)


from experiments.fitting import get_models


from experiments.fitting.utils.ground_truth.gt_dataset import GroundTruthDataset
from experiments.fitting.utils.seed import set_global_determinism
import argparse


def train(space, config_name):
    with initialize(config_path="configs/" + space):
        cfg = compose(config_name=config_name)

    print(cfg)

    assert space == cfg.geometry.input_space.lower()
    assert cfg.data.n_coords % 2 == 0

    # Set device, seed and create log directory
    set_global_determinism(cfg.seed)

    # Create the dataset
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    vmin = min(
        train_loader.dataset.vmin, val_loader.dataset.vmin, test_loader.dataset.vmin
    )
    vmax = max(
        train_loader.dataset.vmax, val_loader.dataset.vmax, test_loader.dataset.vmax
    )

    print(vmin, vmax)

    # Init model
    solver, train_autodecoder, val_autodecoder, test_autodecoder = get_models(
        cfg=cfg,
        vmin=vmin,
        vmax=vmax,
    )

    gt_dict = None
    gt_dataset_val = None
    gt_dataset_test = None
    visualize_gt = None
    visualize_equiv = None
    visualize_reconstructions = None

    if cfg.visualization.active:
        visualize_reconstructions = get_recon_visualization(cfg, vmin, vmax)

    if cfg.eikonal.ground_truth.active:
        gt_dict = get_gt_solver(cfg)

        path_data = f"./experiments/fitting/datasets/{space.lower()}/data/"
        file_name = f"{path_data}coord_{cfg.data.base_dataset.name}"
        if space.lower() == "position_orientation":
            file_name += (
                f"_xi_{cfg.geometry.metric.xi}_epsilon_{cfg.geometry.metric.epsilon}"
            )

        gt_dataset_val = [
            GroundTruthDataset(
                base_dataset=val_loader.dataset.base_dataset,
                solver=gt_dict["solver"],
                name=gt_data["name"],
                grid_data_fn=gt_data["data"],
                cfg=cfg.eikonal.ground_truth,
                precomputed_dir=file_name + "_val",
                save_data=cfg.eikonal.ground_truth.save_data,
                force_recompute=cfg.eikonal.ground_truth.force_recompute,
            )
            for gt_data in gt_dict["grid_data"]
        ]

        gt_dataset_test = [
            GroundTruthDataset(
                base_dataset=test_loader.dataset.base_dataset,
                solver=gt_dict["solver"],
                name=gt_data["name"],
                grid_data_fn=gt_data["data"],
                cfg=cfg.eikonal.ground_truth,
                precomputed_dir=file_name + "_test",
                save_data=cfg.eikonal.ground_truth.save_data,
                force_recompute=cfg.eikonal.ground_truth.force_recompute,
            )
            for gt_data in gt_dict["grid_data"]
        ]

        if cfg.eikonal.ground_truth.num_visualized > 0:
            visualize_gt = get_gt_visualization(cfg, vmin, vmax)

            # check if parameter is true, default is false if does not exist
            if (
                hasattr(cfg.eikonal.ground_truth, "visualize_equiv")
                and cfg.eikonal.ground_truth.visualize_equiv
            ):
                visualize_equiv = get_equiv_visualization(cfg, vmin, vmax)

    aux = (
        cfg.data.base_dataset.name
        + "_"
        + cfg.geometry.input_space
        + "_"
        + cfg.geometry.group
        + "_"
        + str(cfg.geometry.dim_orientation)
    )

    # Initialize wandb
    logger.init(
        name="fitting_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.logging.debug else "online",
    )

    trainer = AutoDecodingEikonalTrainer(
        config=cfg,
        solver=solver,
        train_autodecoder=train_autodecoder,
        val_autodecoder=val_autodecoder,
        test_autodecoder=test_autodecoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        seed=cfg.seed,
        num_epochs=cfg.training.num_epochs,
        visualize_reconstruction=visualize_reconstructions,
        gt_dataset_val=gt_dataset_val,
        gt_dataset_test=gt_dataset_test,
        visualize_gt=visualize_gt,
        visualize_equiv=visualize_equiv,
    )

    trainer.create_functions()

    # Train model
    trainer.train_model()

    # Properly finish wandb run
    logger.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=str, required=True, help="Space")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Config file to load (without .yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional overrides (e.g., database.port=1234)",
    )
    args = parser.parse_args()

    train(args.space, args.experiment)
