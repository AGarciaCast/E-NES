import argparse

from hydra import initialize, compose


import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")


from experiments.fitting.datasets import get_dataloaders

from experiments.fitting.utils.visualization import (
    get_gt_solver,
)


from experiments.fitting.utils.ground_truth.gt_dataset import GroundTruthDataset
from experiments.fitting.utils.seed import set_global_determinism
import argparse


import warnings
import numpy as np
import holoviews as hv  # for visualization


import matplotlib.pyplot as plt


hv.extension("matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def plot_figs(space, config_name):
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

    gt_dict = get_gt_solver(cfg)

    path_data = f"./experiments/fitting/datasets/{space.lower()}/data/"

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

    ds = test_loader.dataset.base_dataset
    gt_ds = gt_dataset_test[0]
    grid_data = gt_ds.grid_data

    x = grid_data["x"]
    y = grid_data["y"]
    Xs = grid_data["Xs"]
    num_ns = Xs.shape

    idx = (num_ns[0] // 2, num_ns[1] // 2)

    for i in range(len(ds)):
        vel = ds[i][0]
        time = gt_ds[i][0][idx]

        # plot images and save them

        # gradients = gradients[..., 2:]

        vmap = hv.Image(
            (x, y, vel.T),
            kdims=["X", "Y"],
            vdims="Velocity",
        ).opts(cmap="viridis", colorbar=False, clim=(vmin, vmax))

        # save image
        fig = vmap.opts(hv.opts.Image(show_legend=False))
        fig = hv.render(fig, backend="matplotlib")
        ax = fig.gca()
        ax.set_axis_off()  # Turn off all axis elements
        ax.set_frame_on(False)  # Remove the frame
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        fig.savefig(f"figures/{cfg.data.base_dataset.name}_velocity_{i}.png", dpi=300)
        plt.close(fig)

        # Your existing code
        max_t = np.max(time)
        min_t = np.min(time)

        # Create an Image element with the jet colormap
        tmapref = hv.Image((x, y, time.T)).opts(cmap="jet", colorbar=False)

        # Create a filled contour plot with automatic level selection
        # Use contours operation with filled=True parameter
        filled_contours = hv.operation.contours(tmapref, filled=True).opts(
            cmap="jet",  # Use the same jet colormap
            alpha=0.7,  # Optional: adjust transparency if needed
        )

        # Add your scatter point
        srcp = hv.Scatter([Xs[idx]]).opts(marker="*", s=200, c="r")

        # Combine the elements
        fig = (filled_contours * srcp).opts(hv.opts.Image(show_legend=False))

        # Render and save
        fig = hv.render(fig, backend="matplotlib")
        ax = fig.gca()
        ax.set_axis_off()  # Turn off all axis elements
        ax.set_frame_on(False)  # Remove the frame
        ax.get_xaxis().set_visible(False)  # Hide x-axis
        ax.get_yaxis().set_visible(False)  # Hide y-axis
        fig.savefig(f"vel_figs/{cfg.data.base_dataset.name}_time_{i}.png", dpi=300)
        plt.close(fig)


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

    plot_figs(args.space, args.experiment)
