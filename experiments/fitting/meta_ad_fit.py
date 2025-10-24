import omegaconf
import argparse

from hydra import initialize, compose


import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

from experiments.fitting.datasets import get_dataloaders
from experiments.fitting.trainers.meta_ad_eikonal_trainer import (
    MetaAutoDecodingEikonalTrainer,
)

from experiments.fitting.utils.visualization import (
    get_recon_visualization,
    get_gt_solver,
    get_gt_visualization,
)


from experiments.fitting import get_models


from experiments.fitting.utils.ground_truth.gt_dataset import GroundTruthDataset
from experiments.fitting.utils.seed import set_global_determinism
from experiments.fitting.utils.logging import logger


import argparse


def train(space, config_name):
    with initialize(config_path="configs/" + space):
        cfg = compose(config_name=config_name)

    print(cfg)

    assert space == cfg.geometry.input_space.lower()
    assert cfg.data.n_coords % 2 == 0
    # meta.num_pairs has to be a multiple of data.num_pairs
    # num_pairs = min(cfg.data.num_pairs, cfg.meta.num_pairs)
    # assert num_pairs % cfg.data.num_pairs == 0

    # Set device, seed and create log directory
    set_global_determinism(cfg.seed)
    # Create the dataset
    temp_num_pairs = cfg.data.num_pairs

    cfg.data.num_pairs = int(cfg.data.n_coords / 2)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, meta=True)

    cfg.data.num_pairs = temp_num_pairs

    vmin = min(
        train_loader.dataset.vmin, val_loader.dataset.vmin, test_loader.dataset.vmin
    )
    vmax = max(
        train_loader.dataset.vmax, val_loader.dataset.vmax, test_loader.dataset.vmax
    )

    # Init model
    solver, train_autodecoder, val_autodecoder, test_autodecoder, init_latents = (
        get_models(cfg=cfg, vmin=vmin, vmax=vmax, meta=True)
    )

    gt_dict = None
    gt_dataset_val = None
    gt_dataset_test = None
    visualize_gt = None
    visualize_reconstructions = None

    if cfg.visualization.active:
        visualize_reconstructions = get_recon_visualization(cfg, vmin, vmax)

    if cfg.eikonal.ground_truth.active:
        gt_dict = get_gt_solver(cfg)

        path_data = f"./experiments/fitting/datasets/{space.lower()}/data/"

        gt_dataset_val = [
            GroundTruthDataset(
                base_dataset=val_loader.dataset.base_dataset,
                solver=gt_dict["solver"],
                name=gt_data["name"],
                grid_data_fn=gt_data["data"],
                cfg=cfg.eikonal.ground_truth,
                precomputed_dir=f"{path_data}coord_{cfg.data.base_dataset.name}_meta_val",
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
                precomputed_dir=f"{path_data}coord_{cfg.data.base_dataset.name}_meta_test",
                save_data=cfg.eikonal.ground_truth.save_data,
                force_recompute=cfg.eikonal.ground_truth.force_recompute,
            )
            for gt_data in gt_dict["grid_data"]
        ]

        if cfg.eikonal.ground_truth.num_visualized > 0:
            visualize_gt = get_gt_visualization(cfg, vmin, vmax)

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
        name="fitting_meta_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.logging.debug else "online",
    )

    trainer = MetaAutoDecodingEikonalTrainer(
        config=cfg,
        solver=solver,
        init_latents=init_latents,
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

    train(args.space, args.experiment + "_meta")
