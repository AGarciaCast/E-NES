import omegaconf
import argparse
import jax


from hydra import initialize, compose

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

from experiments.fitting.utils.logging import logger
from experiments.fitting.datasets import get_dataloaders
from experiments.fitting.trainers.experimental.hybrid_ad_eikonal_trainer import (
    HybridAutoDecodingEikonalTrainer,
)

from experiments.fitting.utils.visualization import (
    get_recon_visualization,
    get_gt_solver,
    get_gt_visualization,
)

from experiments.fitting import get_models
from experiments.fitting.utils.ground_truth.gt_dataset import GroundTruthDataset
from experiments.fitting.utils.seed import set_global_determinism


def train(space, config_name):
    """
    Combined training function that starts with standard autodecoding and then switches to meta-learning.

    Args:
        space: The space configuration (e.g., 'euclidean')
        config_name: The configuration name to load
    """
    # Load configuration
    with initialize(config_path="configs/" + space):
        cfg = compose(config_name=config_name)

    print(f"Loaded configuration: {cfg}")

    # Check configuration validity
    assert space == cfg.geometry.input_space.lower()
    assert cfg.data.n_coords % 2 == 0

    # Set device and seed for reproducibility
    set_global_determinism(cfg.seed)

    # Create dataloaders
    _, val_loader, test_loader = get_dataloaders(cfg)

    # If we are using dataloaders specifically designed for meta-learning, create them
    # First save the original num_pairs setting
    temp_num_pairs = cfg.data.num_pairs

    # Configure for meta-training
    cfg.data.num_pairs = int(cfg.data.n_coords / 2)
    train_loader, _, _ = get_dataloaders(cfg, meta=True)

    # Restore original setting
    cfg.data.num_pairs = temp_num_pairs

    # Get data value ranges
    vmin = min(
        train_loader.dataset.vmin, val_loader.dataset.vmin, test_loader.dataset.vmin
    )
    vmax = max(
        train_loader.dataset.vmax, val_loader.dataset.vmax, test_loader.dataset.vmax
    )

    print(f"Data value range: {vmin} to {vmax}")

    # Initialize models for standard autodecoding
    solver, _, val_autodecoder, test_autodecoder = get_models(
        cfg=cfg,
        vmin=vmin,
        vmax=vmax,
    )

    # Create new models for meta-learning, but we'll reuse the pretrained solver parameters
    # For meta-learning, we only need the solver and init_latents model
    _, train_autodecoder, _, _, init_latents = get_models(
        cfg=cfg, vmin=vmin, vmax=vmax, meta=True
    )

    # Prepare visualization and ground truth evaluation tools
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
                precomputed_dir=f"{path_data}coord_{cfg.data.base_dataset.name}_val",
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
                precomputed_dir=f"{path_data}coord_{cfg.data.base_dataset.name}_test",
                save_data=cfg.eikonal.ground_truth.save_data,
                force_recompute=cfg.eikonal.ground_truth.force_recompute,
            )
            for gt_data in gt_dict["grid_data"]
        ]

        if cfg.eikonal.ground_truth.num_visualized > 0:
            visualize_gt = get_gt_visualization(cfg, vmin, vmax)

    # Create run identifier
    aux = (
        cfg.data.base_dataset.name
        + "_"
        + cfg.geometry.input_space
        + "_"
        + cfg.geometry.group
        + "_"
        + str(cfg.geometry.dim_orientation)
    )

    # Initialize logging for autodecoding phase
    logger.init(
        name="fitting_hybrid_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.logging.debug else "online",
    )

    trainer = HybridAutoDecodingEikonalTrainer(
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

    train(args.space, args.experiment + "_hybrid")
