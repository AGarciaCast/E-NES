import orbax.checkpoint as ocp
import os
from omegaconf import OmegaConf, read_write

import omegaconf


import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

from experiments.fitting.datasets import get_dataloaders
from experiments.fitting.trainers.ad_eikonal_trainer import AutoDecodingEikonalTrainer
from experiments.fitting.trainers.meta_ad_eikonal_trainer import (
    MetaAutoDecodingEikonalTrainer,
)

from experiments.fitting import get_models

from experiments.fitting.utils.seed import set_global_determinism

from experiments.fitting.utils.visualization import (
    get_recon_visualization,
    get_gt_solver,
    get_gt_visualization,
    get_equiv_visualization,
)

from experiments.fitting.utils.ground_truth.gt_dataset import GroundTruthDataset
from experiments.fitting.datasets import get_dataloaders


from typing import Dict, Tuple, Any


def count_params(params) -> int:
    """Count the number of parameters in a parameter tree.

    Args:
        params: JAX parameter tree (nested dictionaries/lists containing arrays)

    Returns:
        Total number of parameters
    """
    # Flatten the parameter tree to get all leaves (parameter arrays)
    flat_params = jax.tree_util.tree_leaves(params)

    # Count the number of elements in each array and sum them up
    total_params = sum(x.size for x in flat_params)

    return total_params


def count_model_params(train_state) -> Tuple[int, int]:
    """Count parameters for solver and autodecoder separately.

    Args:
        train_state: TrainState object containing model parameters

    Returns:
        Tuple of (solver_count, autodecoder_count)
    """
    solver_params = train_state.params["solver"]
    autodecoder_params = train_state.params["autodecoder"]

    solver_count = count_params(solver_params)
    autodecoder_count = count_params(autodecoder_params)

    print(f"Solver parameters: {solver_count}")
    print(f"Autodecoder parameters: {autodecoder_count}")
    print(f"Total parameters: {solver_count + autodecoder_count}")

    return solver_count, autodecoder_count


def autodecoding_import(
    wandb_ref, aux_out=False, gt_val=False, plot=False, train=False, num_epochs_auto=-1
):

    checkpoint_dir = f"./checkpoints/{wandb_ref}"
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    checkpoint_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=1,
    )
    checkpoint_manager = ocp.CheckpointManager(
        directory=checkpoint_dir,
        options=checkpoint_options,
        item_handlers={
            "state": ocp.StandardCheckpointHandler(),
            "config": ocp.JsonCheckpointHandler(),
        },
        item_names=["state", "config"],
    )
    ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())
    cfg = OmegaConf.create(ckpt.config)

    cfg.eikonal.ground_truth.force_recompute = False
    cfg.data.train_force_recompute = False
    cfg.data.val_force_recompute = False
    cfg.data.test_force_recompute = False

    if "meta" in list(ckpt.config.keys()):
        return meta_autodecoding_import(
            cfg,
            ckpt.state,
            aux_out=aux_out,
        )
    else:
        return vanilla_autodecoding_import(
            cfg,
            ckpt.state,
            aux_out=aux_out,
            train=train,
            num_epochs_auto=num_epochs_auto,
        )


def meta_autodecoding_import(cfg, state_dict, aux_out=False):
    from experiments.fitting.utils.logging import logger

    # Set device, seed and create log directory
    set_global_determinism(cfg.seed)
    cfg.data.train_force_recompute = False
    cfg.data.val_force_recompute = False
    cfg.data.test_force_recompute = False
    cfg.eikonal.ground_truth.force_recompute = False

    # Create the dataset
    temp_num_pairs = cfg.data.num_pairs

    cfg.data.num_pairs = int(cfg.data.n_coords / 2)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, meta=True)

    cfg.data.num_pairs = temp_num_pairs

    try:
        vmin = state_dict["vmin"]
        vmax = state_dict["vmax"]
    except:
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
        name="downstream_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode="disabled",
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
    state = trainer.TrainState(**state_dict)

    _, _, _, test_autodecoder_global = get_models(cfg=cfg, vmin=vmin, vmax=vmax)

    key = jax.random.PRNGKey(cfg.seed)

    # Split key
    key, autodecoder_key = jax.random.split(key)

    autodecoder_params = test_autodecoder_global.init(
        autodecoder_key,
        jnp.ones(cfg.data.test_batch_size, dtype=jnp.int32),
    )

    # Process each batch in the evaluation loader
    for batch_idx, batch in enumerate(test_loader):
        # Measure fitting time
        loss, mse, (state, inner_state) = trainer.test_step(state, batch)
        _, _, vel_idx = batch
        autodecoder_params["params"]["pose_pos"] = (
            autodecoder_params["params"]["pose_pos"]
            .at[vel_idx]
            .set(inner_state.params["autodecoder"]["params"]["pose_pos"])
        )

        autodecoder_params["params"]["appearance"] = (
            autodecoder_params["params"]["appearance"]
            .at[vel_idx]
            .set(inner_state.params["autodecoder"]["params"]["appearance"])
        )

        if autodecoder_params["params"]["pose_ori"] is not None:
            autodecoder_params["params"]["pose_ori"] = (
                autodecoder_params["params"]["pose_ori"]
                .at[vel_idx]
                .set(inner_state.params["autodecoder"]["params"]["pose_ori"])
            )

    solver_params = state.params["solver"]
    logger.finish()

    if aux_out:
        return (
            solver,
            solver_params,
            test_autodecoder_global,
            autodecoder_params,
            test_loader,
            (vmin, vmax),
            (cfg.data.x_min, cfg.data.x_max),
        )
    else:
        return (
            solver,
            solver_params,
            test_autodecoder_global,
            autodecoder_params,
            test_loader,
        )


def vanilla_autodecoding_import(
    cfg,
    state_dict,
    aux_out=False,
    train=False,
    num_epochs_auto=-1,
):
    from experiments.fitting.utils.logging import logger

    set_global_determinism(cfg.seed)
    cfg.data.train_force_recompute = False
    cfg.data.val_force_recompute = False
    cfg.data.test_force_recompute = False
    cfg.eikonal.ground_truth.force_recompute = False

    # Create the dataset
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    try:
        vmin = state_dict["vmin"]
        vmax = state_dict["vmax"]
    except:
        vmin = min(
            train_loader.dataset.vmin, val_loader.dataset.vmin, test_loader.dataset.vmin
        )
        vmax = max(
            train_loader.dataset.vmax, val_loader.dataset.vmax, test_loader.dataset.vmax
        )

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
    visualize_reconstructions = None

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
        name="downstream_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode="disabled",
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
    state = trainer.TrainState(**state_dict)

    if not train:
        if num_epochs_auto > 0:
            state = state.replace(autodecoder_steps=num_epochs_auto)

        test_state = trainer.fit_autodecoding(state, name="test")
        solver_params = test_state.params["solver"]
        autodecoder_params = test_state.params["autodecoder"]
        autodecoder = test_autodecoder
        loader = test_loader

    else:
        solver_params = state.params["solver"]
        autodecoder_params = state.params["autodecoder"]
        autodecoder = train_autodecoder
        loader = train_loader

    logger.finish()

    if aux_out:
        return (
            solver,
            solver_params,
            autodecoder,
            autodecoder_params,
            loader,
            (vmin, vmax),
            (cfg.data.x_min, cfg.data.x_max),
        )
    else:
        return (
            solver,
            solver_params,
            autodecoder,
            autodecoder_params,
            loader,
        )


def vanilla_full_test_import(
    cfg,
    state_dict,
    num_epochs_auto=-1,
    gt_val=False,
    plot=False,
    skip_r=-1,
    skip_s=-1,
    force_recompute=False,
    activate_gt=None,
    warmup=False,
    mul_batch=1,
    plot_equiv=False,
    num_visualised=-1,
    final_plot_gt=True,
    final_plot_equiv=False,
):
    from experiments.fitting.utils.logging import logger

    set_global_determinism(cfg.seed)

    # Change skip_r skip_s, for eval
    with read_write(cfg):
        cfg.data.train_force_recompute = False
        cfg.data.val_force_recompute = False
        cfg.data.test_force_recompute = False
        if num_visualised > 0:
            cfg.eikonal.ground_truth.num_visualized = num_visualised
        cfg.eikonal.ground_truth.force_recompute = force_recompute
        if activate_gt is not None:
            cfg.eikonal.ground_truth.active = activate_gt
        if skip_r > 1:
            cfg.eikonal.ground_truth.skip_r = skip_r
        if skip_s > 1:
            cfg.eikonal.ground_truth.skip_s = skip_s

    # Create the dataset
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    try:
        vmin = state_dict["vmin"]
        vmax = state_dict["vmax"]
    except:
        vmin = min(
            train_loader.dataset.vmin, val_loader.dataset.vmin, test_loader.dataset.vmin
        )
        vmax = max(
            train_loader.dataset.vmax, val_loader.dataset.vmax, test_loader.dataset.vmax
        )

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
    visualize_reconstructions = None
    visualize_equiv = None

    if plot and cfg.visualization.active:
        visualize_reconstructions = get_recon_visualization(cfg, vmin, vmax)

    if gt_val and cfg.eikonal.ground_truth.active:
        gt_dict = get_gt_solver(cfg)

        path_data = (
            f"./experiments/fitting/datasets/{cfg.geometry.input_space.lower()}/data/"
        )

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

        if plot and cfg.eikonal.ground_truth.num_visualized > 0:
            visualize_gt = get_gt_visualization(
                cfg,
                vmin,
                vmax,
                final=final_plot_gt,
            )

            # check if parameter is true, default is false if does not exist
            if plot_equiv:
                visualize_equiv = get_equiv_visualization(
                    cfg, vmin, vmax, final=final_plot_equiv
                )

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

    if num_epochs_auto > 0:
        aux = f"{num_epochs_auto}_steps_" + aux

    logger.init(
        name=f"testing_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
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
    state = trainer.TrainState(**state_dict)

    trainer.num_pairs_batch_test = trainer.num_pairs_batch_test * mul_batch

    if num_epochs_auto > 0:
        state = state.replace(autodecoder_steps=num_epochs_auto)

    count_model_params(state)
    print(state.params["autodecoder"]["params"]["pose_pos"].shape)

    # Perform a full test epoch
    if warmup:
        trainer.test_epoch(state)

    trainer.test_epoch(state)

    test_metrics = {k: v for k, v in trainer.metrics.items() if "test" in k}
    logger.finish()

    return test_metrics


def vanilla_full_test_import_meta(
    cfg,
    state_dict,
    gt_val=False,
    plot=False,
    skip_r=-1,
    skip_s=-1,
    force_recompute=False,
    activate_gt=None,
    warmup=False,
    mul_batch=1,
):
    from experiments.fitting.utils.logging import logger

    # Set device, seed and create log directory
    set_global_determinism(cfg.seed)

    with read_write(cfg):
        cfg.data.train_force_recompute = False
        cfg.data.val_force_recompute = False
        cfg.data.test_force_recompute = False
        cfg.eikonal.ground_truth.force_recompute = force_recompute
        if activate_gt is not None:
            cfg.eikonal.ground_truth.active = activate_gt
        if skip_r > 1:
            cfg.eikonal.ground_truth.skip_r = skip_r
        if skip_s > 1:
            cfg.eikonal.ground_truth.skip_s = skip_s

    # Create the dataset
    temp_num_pairs = cfg.data.num_pairs

    cfg.data.num_pairs = int(cfg.data.n_coords / 2)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, meta=True)

    cfg.data.num_pairs = temp_num_pairs

    try:
        vmin = state_dict["vmin"]
        vmax = state_dict["vmax"]
    except:
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

    if plot and cfg.visualization.active:
        visualize_reconstructions = get_recon_visualization(cfg, vmin, vmax)

    if gt_val and cfg.eikonal.ground_truth.active:
        gt_dict = get_gt_solver(cfg)

        path_data = (
            f"./experiments/fitting/datasets/{cfg.geometry.input_space.lower()}/data/"
        )

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

        if plot and cfg.eikonal.ground_truth.num_visualized > 0:
            visualize_gt = get_gt_visualization(cfg, vmin, vmax, final=True)

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
        name="testing_meta_" + aux,
        dir=cfg.logging.log_dir,
        project="EquivEikonal",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
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
    state = trainer.TrainState(**state_dict)

    trainer.num_pairs_batch_test = trainer.num_pairs_batch_test * mul_batch
    # Perform a full test epoch
    if warmup:
        trainer.test_epoch(state)
    trainer.test_epoch(state)

    test_metrics = {k: v for k, v in trainer.metrics.items() if "test" in k}
    logger.finish()

    return test_metrics


if __name__ == "__main__":
    autodecoding_import("kujk5zmb")
