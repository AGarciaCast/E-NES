from pathlib import Path

import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from experiments.downstream.utils.autodecoding_import import (
    vanilla_full_test_import,
    vanilla_full_test_import_meta,
)


checkpoint_dir = str(Path("./checkpoints/76p75077").absolute())


"""
35 => 2
24 => 3
18 => 4
14 => 5
12=> 6
"""

aux_a = [0] * 70

for skip_s in [35, 24, 18, 14, 12]:
    # Load the checkpoint
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

    # Config adjustments
    cfg = OmegaConf.create(ckpt.config)
    cfg.data.base_dataset.name = cfg.data.base_dataset.name.split("-3d")[0]

    test_metrics = vanilla_full_test_import(
        cfg=cfg,
        state_dict=ckpt.state,
        skip_s=skip_s,
        num_epochs_auto=100,
        force_recompute=True,
        activate_gt=True,
        warmup=True,
        gt_val=True,
        mul_batch=350,
    )

    print(
        f"Saved test metrics for grid {len(aux_a[::skip_s])}x{len(aux_a[::skip_s])}x{len(aux_a[::skip_s])}:",
        test_metrics,
    )


checkpoint_dir = str(Path("./checkpoints/y6f8n4mt").absolute())


print("-" * 10 + "META" + "-" * 10)

for skip_s in [35, 24, 18, 14, 12]:

    # Load the checkpoint
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

    # Config adjustments
    cfg = OmegaConf.create(ckpt.config)
    cfg.data.base_dataset.name = cfg.data.base_dataset.name.split("-3d")[0]
    test_metrics = vanilla_full_test_import_meta(
        cfg=cfg,
        state_dict=ckpt.state,
        skip_s=skip_s,
        force_recompute=True,
        activate_gt=True,
        warmup=True,
        gt_val=True,
        mul_batch=350,
    )

    print(
        f"Saved test metrics for grid {len(aux_a[::skip_s])}x{len(aux_a[::skip_s])}x{len(aux_a[::skip_s])}:",
        test_metrics,
    )
