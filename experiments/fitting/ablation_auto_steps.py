import orbax.checkpoint as ocp
import os
from omegaconf import OmegaConf

import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

from experiments.downstream.utils.autodecoding_import import vanilla_full_test_import

import pandas as pd


def run_ablation(
    wandb_ref,
    num_steps,
    plot=-1,
    save=False,
    warmup=False,
    equiv_plot=False,
    num_visualised=-1,
    final_plot_gt=True,
    final_plot_equiv=False,
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
        raise ValueError()

    results = []
    for steps in num_steps:

        outputs = vanilla_full_test_import(
            cfg=cfg,
            state_dict=ckpt.state,
            num_epochs_auto=steps,
            gt_val=True,
            plot=plot == steps,
            warmup=warmup or save,
            plot_equiv=equiv_plot,
            num_visualised=num_visualised if plot == steps else -1,
            final_plot_gt=final_plot_gt,
            final_plot_equiv=final_plot_equiv,
        )

        test_metrics = outputs

        test_metrics["steps"] = steps
        results.append(test_metrics)

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(results)

    # Display the DataFrame
    print(df)

    # Save the DataFrame to a CSV file
    name_ds = cfg.data.base_dataset.name
    if save:
        csv_filename = f"./experiments/fitting/results/results_{name_ds}.csv"
        df.to_csv(csv_filename, index=False)

    return df, name_ds


if __name__ == "__main__":

    num_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # Flat A
    df, name_ds = run_ablation(
        wandb_ref="9ybqwgwn", num_steps=num_steps, save=True, plot=500, warmup=True
    )

    # Flat B
    df, name_ds = run_ablation(
        wandb_ref="kujk5zmb", num_steps=num_steps, save=True, plot=450, warmup=True
    )

    # Curve A
    df, name_ds = run_ablation(
        wandb_ref="jke7y380", num_steps=num_steps, save=True, plot=450, warmup=True
    )

    # Curve B
    df, name_ds = run_ablation(
        wandb_ref="kv3v8k2t", num_steps=num_steps, save=True, plot=500, warmup=True
    )

    # FlatFault A
    df, name_ds = run_ablation(
        wandb_ref="n8d7gbvh", num_steps=num_steps, save=True, plot=450, warmup=True
    )

    # FlatFault B
    df, name_ds = run_ablation(
        wandb_ref="aeur089d", num_steps=num_steps, save=True, plot=500, warmup=True
    )

    # CurveFault A
    df, name_ds = run_ablation(
        wandb_ref="7fv5f58o", num_steps=num_steps, save=True, plot=500, warmup=True
    )

    # CurveFault B
    df, name_ds = run_ablation(
        wandb_ref="l5dvsxhz", num_steps=num_steps, save=True, plot=400, warmup=True
    )

    # Style A
    df, name_ds = run_ablation(
        wandb_ref="je82gqf7", num_steps=num_steps, save=True, plot=500, warmup=True
    )

    # Style B
    df, name_ds = run_ablation(
        wandb_ref="2164zvsr", num_steps=num_steps, save=True, plot=400, warmup=True
    )
