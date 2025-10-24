import omegaconf
import argparse

from hydra import initialize, compose

from experiments.fitting.utils.logging import logger

from experiments.fitting.datasets import get_dataloaders
from experiments.fitting.trainers.ad_eikonal_trainer import AutoDecodingEikonalTrainer

from experiments.fitting.utils.visualization import (
    get_recon_visualization,
    get_gt_solver,
    get_gt_visualization,
)


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
    from equiv_eikonal.models.solvers.euclidean_solver import (
        EuclideanFunctaNeuralEikonalSolver,
    )

    solver = EuclideanFunctaNeuralEikonalSolver(
        num_hidden=cfg.solver.num_hidden,
        latent_dim=cfg.geometry.latent_dim,
        embedding_freq_multiplier=cfg.solver.embedding_freq_multiplier_invariant,
        vmin=vmin,
        vmax=vmax,
        factored=cfg.eikonal.factored,
        invariant=None,
        num_heads=None,
        embedding_type=None,
    )

    # Init autodecoders
    from equiv_eikonal.latents.vanilla_affine_orthogonal import FunctaLatents

    train_autodecoder = FunctaLatents(
        num_signals=len(train_loader.dataset),
        dim_signal=cfg.geometry.dim_signal,
        dim_orientation=cfg.geometry.dim_orientation,
        latent_dim=cfg.geometry.latent_dim,
        num_latents=cfg.geometry.num_latents,
    )
    val_autodecoder = FunctaLatents(
        num_signals=len(val_loader.dataset),
        dim_signal=cfg.geometry.dim_signal,
        dim_orientation=cfg.geometry.dim_orientation,
        latent_dim=cfg.geometry.latent_dim,
        num_latents=cfg.geometry.num_latents,
    )
    test_autodecoder = FunctaLatents(
        num_signals=len(test_loader.dataset),
        dim_signal=cfg.geometry.dim_signal,
        dim_orientation=cfg.geometry.dim_orientation,
        latent_dim=cfg.geometry.latent_dim,
        num_latents=cfg.geometry.num_latents,
    )

    # solver, train_autodecoder, val_autodecoder, test_autodecoder = get_models(
    #     cfg=cfg,
    #     vmin=vmin,
    #     vmax=vmax,
    # )

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
