from experiments.fitting.utils.logging import logger  # Add this line
import warnings
import numpy as np
import holoviews as hv  # for visualization
import jax.numpy as jnp

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image

import matplotlib.pyplot as plt

hv.extension("matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def visualize_gt_euclidean_2D(
    grid_data,
    vel,
    pred_vel,
    ref_times,
    pred_times,
    gradients,
    name,
    vmin=0.0,
    vmax=1.0,
    sp=2,
    all_indices=False,
    final_path=None,
):
    x = grid_data["x"]
    y = grid_data["y"]
    Xs = grid_data["Xs"]
    num_ns = Xs.shape
    gradients = gradients[..., 2:]

    if all_indices:
        indices = [(i, j) for i in range(num_ns[0]) for j in range(num_ns[1])]
    else:
        indices = [
            (1, 1),
            (num_ns[0] // 2, num_ns[1] // 2),
            (num_ns[0] - 2, num_ns[1] - 2),
        ]

    for ixs in indices:

        if final_path is None:

            vmap = hv.Image(
                (x, y, vel.T), kdims=["X", "Y"], vdims="Velocity", label="V, "
            ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

            sample_vel = pred_vel[ixs][..., 1]

            rec_vmap = hv.Image(
                (x, y, sample_vel.T), kdims=["X", "Y"], vdims="Velocity", label="V, "
            ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

            colors = ["black", "white"]
            max_t = max(np.max(ref_times[ixs]), np.max(pred_times[ixs]))
            min_t = min(np.min(ref_times[ixs]), np.min(pred_times[ixs]))
            tmapref = hv.Image((x, y, ref_times[ixs].T), label="T_ref")
            tmap = hv.Image((x, y, pred_times[ixs].T), label="T_pred")

            levels = np.linspace(ref_times[ixs].min(), ref_times[ixs].max(), 15)

            tctrref = hv.operation.contours(tmapref, levels=levels).opts(
                color=colors[0], cmap=[colors[0]], linestyle="solid", linewidth=4
            )
            tctr = hv.operation.contours(tmap, levels=levels).opts(
                color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
            )

            srcp = hv.Scatter([Xs[ixs]]).opts(marker="*", s=200, c="r")

            gradients_ = gradients[ixs][::sp, ::sp]
            mag = np.linalg.norm(gradients_, axis=-1) * 0.05
            angle = (np.pi / 2.0) - np.arctan2(
                gradients_[..., 0] / mag, gradients_[..., 1] / mag
            )
            # angle = -np.arctan2(gradients_[..., 1]/mag, gradients_[..., 0]/mag)
            vf = (
                hv.VectorField((x[::sp], y[::sp], angle.T, mag.T))
                .opts(magnitude="Magnitude")
                .opts(hv.opts.VectorField(pivot="tail"))
            )

            fig1 = (
                (vmap * tctrref * tctr * srcp)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title="Solution contours")
            )

            fig2 = (
                (vmap * srcp * vf)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title="Gradients")
            )

            tmapref = tmapref.opts(
                cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8)
            )
            tmap = tmap.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))

            aux_data = np.abs(ref_times[ixs].T - pred_times[ixs].T)

            tmapdiff = hv.Image((x, y, aux_data), label="T_ref").opts(
                cmap="jet",
                colorbar=True,
                clim=(np.min(aux_data), np.max(aux_data) + 1e-8),
            )

            fig3 = (
                (tmapref * srcp)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title="Ground Truth Time")
            )

            fig4 = (
                (tmap * srcp)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title="Pred Time")
            )

            fig5 = (
                (tmapdiff)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title="Diff Time")
            )

            fig = (
                (rec_vmap * srcp).opts(title="Recon vel")
                + fig1
                + fig2
                + fig3
                + fig4
                + fig5
            ).cols(3)
            fig = hv.render(fig, backend="matplotlib")

            # MODIFIED: Use logger.log_image instead of direct wandb call
            logger.log_image(f"{name}_{ixs}", fig)
            plt.close(fig)

        else:
            # Set paper-ready matplotlib parameters
            plt.rcParams.update(
                {
                    # Use serif fonts - NeurIPS uses Times Roman (ptm)
                    "font.family": "serif",
                    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                    "mathtext.fontset": "cm",
                    # Font sizes
                    "font.size": 14,  # Regular text
                    "axes.titlesize": 18,  # Title size
                    "axes.labelsize": 14,  # Axis label size
                    "xtick.labelsize": 14,  # X tick label size
                    "ytick.labelsize": 14,  # Y tick label size
                    "legend.fontsize": 12,  # Legend font size
                    # Line widths
                    "axes.linewidth": 0.5,
                    "grid.linewidth": 0.5,
                    "lines.linewidth": 1.0,
                    "lines.markersize": 3,
                    # Clean style for academic publications
                    "axes.grid": False,
                    "axes.facecolor": "white",
                    "axes.edgecolor": "black",
                    "grid.color": "#CCCCCC",
                    "grid.linestyle": "--",
                    # Legend settings
                    "legend.frameon": False,
                    "legend.numpoints": 1,
                    "legend.handlelength": 2,
                    # Use TrueType fonts for better PDF output
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                }
            )

            # Create a 1x4 figure with controlled spacing for final=True case
            fig, axes = plt.subplots(
                1, 4, figsize=(16, 4)
            )  # Adjusted figure size for better proportions

            # Add more space between subplots
            plt.subplots_adjust(wspace=0.4)  # Increase spacing between subplots

            max_t = max(np.max(ref_times[ixs]), np.max(pred_times[ixs]))
            min_t = min(np.min(ref_times[ixs]), np.min(pred_times[ixs]))

            # Clear the existing axes for fresh plots
            for ax in axes.flat:
                ax.clear()

            # Velocity plot (first)
            im0 = axes[0].imshow(
                vel.T,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[0].plot(Xs[ixs][0], Xs[ixs][1], "r*", markersize=20)
            axes[0].set_title("Velocity Field")
            # Format colorbar to be smaller and thinner
            cbar0 = fig.colorbar(
                im0,
                ax=axes[0],
                shrink=0.83,  # Make colorbar 80% of axis height
                pad=0.02,  # Reduce padding between plot and colorbar
                aspect=20,  # Make colorbar thinner
            )
            cbar0.ax.tick_params(labelsize=12)  # Smaller tick labels

            # Ground Truth Time (second)
            im1 = axes[1].imshow(
                ref_times[ixs].T,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="jet",
                vmin=min_t,
                vmax=max_t,
            )
            axes[1].plot(Xs[ixs][0], Xs[ixs][1], "r*", markersize=20)
            axes[1].set_title("Ground Truth Time")
            cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.83, pad=0.02, aspect=20)
            cbar1.ax.tick_params(labelsize=12)

            # Predicted Time (third)
            im2 = axes[2].imshow(
                pred_times[ixs].T,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="jet",
                vmin=min_t,
                vmax=max_t,
            )
            axes[2].plot(Xs[ixs][0], Xs[ixs][1], "r*", markersize=20)
            axes[2].set_title("Predicted Time")
            cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.83, pad=0.02, aspect=20)
            cbar2.ax.tick_params(labelsize=12)

            # Difference (fourth)
            aux_data = np.abs(ref_times[ixs].T - pred_times[ixs].T)
            im3 = axes[3].imshow(
                aux_data,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="jet",
            )
            axes[3].set_title("Absolute Error")
            cbar3 = fig.colorbar(im3, ax=axes[3], shrink=0.83, pad=0.02, aspect=20)
            cbar3.ax.tick_params(labelsize=12)

            # Make plots more square by adjusting aspect ratio
            for ax in axes:
                ax.set_aspect("equal")
                # Optional: remove ticks to save space
                ax.set_xticks([])
                ax.set_yticks([])

            # Save the figure - modified to use tight_layout without rect parameter first
            plt.tight_layout()

            # Save the figure
            plt.savefig(final_path + f"/{name}_{ixs}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)


def visualize_equivariance_euclidean_2D(
    vel,
    pred_vel,
    ref_times,
    pred_times,
    name,
    solver_fn,
    solver_params,
    p,
    a,
    vmin=0.0,
    vmax=1.0,
    x_min=None,
    x_max=None,
    final_path=None,
):
    """Visualize the reconstructions and the poses.

    Args:
        targets: The ground truth images.
        a: The latent features. [batch_size, num_latents, latent_dim]
        p: The latent poses. [batch_size, num_latents, num_in]
        window: The latent window. [batch_size, num_latents, 1]
        image_shape: Tuple of the image shape.
        name: The name of the visualization.
        seed: Random seed for reproducibility.
    """
    p_pos, p_ori = p

    xmin, ymin = x_min if x_min is not None else [-1.0, -1.0]
    xmax, ymax = x_max if x_max is not None else [1.0, 1.0]

    nx, ny = vel.shape
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    xr = x
    yr = y

    skip_s = 5
    xs = x[::skip_s]
    ys = y[::skip_s]

    Xs = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
    num_ns = Xs.shape

    coords = np.stack(np.meshgrid(xs, ys, xr, yr, indexing="ij"), axis=-1)[
        (num_ns[0] // 2, num_ns[1] // 2)
    ].reshape(1, -1, 2, 2)

    ############################################################################################################
    # Rotate poses
    ############################################################################################################
    # Create rotation matrices for 45, 90 and 180 degrees
    theta_45 = np.pi / 4  # 45 degrees
    theta_90 = np.pi / 2  # 90 degrees
    theta_180 = np.pi  # 180 degrees

    R_45 = jnp.array(
        [[np.cos(theta_45), np.sin(theta_45)], [-np.sin(theta_45), np.cos(theta_45)]],
        dtype=p_pos.dtype,
    )

    R_90 = jnp.array(
        [[np.cos(theta_90), np.sin(theta_90)], [-np.sin(theta_90), np.cos(theta_90)]],
        dtype=p_pos.dtype,
    )

    R_180 = jnp.array(
        [
            [np.cos(theta_180), np.sin(theta_180)],
            [-np.sin(theta_180), np.cos(theta_180)],
        ],
        dtype=p_pos.dtype,
    )

    # JAX arrays are immutable, so we create new arrays instead of cloning
    # p_pos_rotated_45 = p_pos @ R_45  # Rotate by 45 degrees
    # p_pos_rotated_90 = p_pos @ R_90  # Rotate by 90 degrees
    # p_pos_rotated_180 = p_pos @ R_180  # Rotate by 180 degrees

    p_pos_rotated_45 = jnp.einsum("ij,bnj->bni", R_45, p_pos)
    p_pos_rotated_90 = jnp.einsum("ij,bnj->bni", R_90, p_pos)
    p_pos_rotated_180 = jnp.einsum("ij,bnj->bni", R_180, p_pos)

    if p_ori is not None:
        p_ori_rotated_45 = jnp.einsum("ij,bnjk->bnik", R_45, p_ori)
        p_ori_rotated_90 = jnp.einsum("ij,bnjk->bnik", R_90, p_ori)
        p_ori_rotated_180 = jnp.einsum("ij,bnjk->bnik", R_180, p_ori)
        # p_ori_rotated_45 = p_ori @ R_45
        # p_ori_rotated_90 = p_ori @ R_90
        # p_ori_rotated_180 = p_ori @ R_180
        p_rotated_45 = (p_pos_rotated_45, p_ori_rotated_45)
        p_rotated_90 = (p_pos_rotated_90, p_ori_rotated_90)
        p_rotated_180 = (p_pos_rotated_180, p_ori_rotated_180)
    else:
        p_rotated_45 = (p_pos_rotated_45, None)
        p_rotated_90 = (p_pos_rotated_90, None)
        p_rotated_180 = (p_pos_rotated_180, None)

    ############################################################################################################
    # Translate poses
    ############################################################################################################
    # stepsize
    step_y = (ymax - ymin) / (ny - 1)

    translation = jnp.asarray([0.0, 10 * step_y], dtype=p_pos.dtype)

    p_pos_translated = p_pos + translation
    p_translated = (p_pos_translated, p_ori)
    # print(solver_fn(solver_params, inputs=coords, p=p_rotated_90, a=a).shape, coords.shape)

    # Get the outputs for the batch, and its rotated and translated versions
    outputs_rot_45 = solver_fn(solver_params, inputs=coords, p=p_rotated_45, a=a)
    outputs_rot_90 = solver_fn(solver_params, inputs=coords, p=p_rotated_90, a=a)
    outputs_rot_180 = solver_fn(solver_params, inputs=coords, p=p_rotated_180, a=a)
    outputs_translated = solver_fn(solver_params, inputs=coords, p=p_translated, a=a)

    # Check if ref_times has the 4D meshgrid structure (xs, ys, xr, yr)
    # If so, we need to extract the central slice using indices based on actual shape
    if len(ref_times.shape) == 4:
        # Use the actual shape of ref_times to compute the center indices
        idx0 = ref_times.shape[0] // 2
        idx1 = ref_times.shape[1] // 2
        ref_times = np.array(ref_times[idx0, idx1].reshape(nx, ny))
        pred_times = np.array(pred_times[idx0, idx1].reshape(nx, ny))
        rec_vel = np.array(pred_vel[idx0, idx1][..., 1].reshape(nx, ny))
    else:
        # Otherwise they're already in the right shape
        ref_times = np.array(ref_times)
        pred_times = np.array(pred_times)
        rec_vel = np.array(pred_vel[..., 1])

    pred_times_rot_45 = np.array(outputs_rot_45[0].reshape(nx, ny))
    rec_vel_rot_45 = np.array(outputs_rot_45[2][..., 1].reshape(nx, ny))

    pred_times_rot_90 = np.array(outputs_rot_90[0].reshape(nx, ny))
    rec_vel_rot_90 = np.array(outputs_rot_90[2][..., 1].reshape(nx, ny))

    pred_times_rot_180 = np.array(outputs_rot_180[0].reshape(nx, ny))
    rec_vel_rot_180 = np.array(outputs_rot_180[2][..., 1].reshape(nx, ny))

    pred_times_translated = np.array(outputs_translated[0].reshape(nx, ny))
    rec_vel_translated = np.array(outputs_translated[2][..., 1].reshape(nx, ny))

    if final_path is None:
        vmap = hv.Image(
            (x, y, vel.T), kdims=["X", "Y"], vdims="Velocity", label="V, "
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        rec_vmap = hv.Image(
            (x, y, rec_vel.T), kdims=["X", "Y"], vdims="Velocity", label="V, "
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        rec_vmap_rot_45 = hv.Image(
            (x, y, rec_vel_rot_45.T),
            kdims=["X", "Y"],
            vdims="Velocity",
            label="V_rot_45, ",
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        rec_vmap_rot_90 = hv.Image(
            (x, y, rec_vel_rot_90.T),
            kdims=["X", "Y"],
            vdims="Velocity",
            label="V_rot_90, ",
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        rec_vmap_rot_180 = hv.Image(
            (x, y, rec_vel_rot_180.T),
            kdims=["X", "Y"],
            vdims="Velocity",
            label="V_rot_180, ",
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        rec_vmap_translated = hv.Image(
            (x, y, rec_vel_translated.T),
            kdims=["X", "Y"],
            vdims="Velocity",
            label="V_translated, ",
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        colors = ["black", "black"]
        max_t = max(
            np.max(ref_times),
            np.max(pred_times),
            np.max(pred_times_rot_45),
            np.max(pred_times_rot_90),
            np.max(pred_times_rot_180),
            np.max(pred_times_translated),
        )
        min_t = min(
            np.min(ref_times),
            np.min(pred_times),
            np.min(pred_times_rot_45),
            np.min(pred_times_rot_90),
            np.min(pred_times_rot_180),
            np.min(pred_times_translated),
        )

        tmap_ref = hv.Image((x, y, ref_times.T), label="T_ref")
        tmap = hv.Image((x, y, pred_times.T), label="T_pred")
        tmap_rot_45 = hv.Image((x, y, pred_times_rot_45.T), label="T_pred_rot_45")
        tmap_rot_90 = hv.Image((x, y, pred_times_rot_90.T), label="T_pred_rot_90")
        tmap_rot_180 = hv.Image((x, y, pred_times_rot_180.T), label="T_pred_rot_180")
        tmap_translated = hv.Image(
            (x, y, pred_times_translated.T), label="T_pred_translated"
        )

        tmap_ref = tmap_ref.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))
        tmap = tmap.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))
        tmap_rot_45 = tmap_rot_45.opts(
            cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8)
        )
        tmap_rot_90 = tmap_rot_90.opts(
            cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8)
        )
        tmap_rot_180 = tmap_rot_180.opts(
            cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8)
        )
        tmap_translated = tmap_translated.opts(
            cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8)
        )

        levels = np.linspace(ref_times.min(), ref_times.max(), 15)

        tctr_ref = hv.operation.contours(tmap_ref, levels=levels).opts(
            color=colors[0], cmap=[colors[0]], linestyle="dashed", linewidth=2
        )

        tctr = hv.operation.contours(tmap, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
        )

        tctr_rot_45 = hv.operation.contours(tmap_rot_45, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
        )

        tctr_rot_90 = hv.operation.contours(tmap_rot_90, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
        )

        tctr_rot_180 = hv.operation.contours(tmap_rot_180, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
        )

        tctr_translated = hv.operation.contours(tmap_translated, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
        )

        fig0 = (tmap_ref * tctr_ref).opts(title="Reference Time")
        fig1 = (tmap * tctr).opts(title="Predicted Time")
        fig2 = (tmap_rot_45 * tctr_rot_45).opts(title="Pred. Time Rotated 45°")
        fig3 = (tmap_rot_90 * tctr_rot_90).opts(title="Pred. Time Rotated 90°")
        fig4 = (tmap_rot_180 * tctr_rot_180).opts(title="Pred. Time Rotated 180°")
        fig5 = (tmap_translated * tctr_translated).opts(title="Pred. Time Translated")
        fig6 = vmap.opts(title="Original Vel")
        fig7 = rec_vmap.opts(title="Recon Vel")
        fig8 = rec_vmap_rot_45.opts(title="Recon Vel Rot 45°")
        fig9 = rec_vmap_rot_90.opts(title="Recon Vel Rot 90°")
        fig10 = rec_vmap_rot_180.opts(title="Recon Vel Rot 180°")
        fig11 = rec_vmap_translated.opts(title="Recon Vel Translated")

        fig = (
            fig0
            + fig1
            + fig2
            + fig3
            + fig4
            + fig5
            + fig6
            + fig7
            + fig8
            + fig9
            + fig10
            + fig11
        ).cols(6)
        fig = hv.render(fig, backend="matplotlib")

        # MODIFIED: Use logger.log_image instead of direct wandb call
        logger.log_image(f"{name}_equiv", fig)
        plt.close(fig)

    else:
        # Camera-ready matplotlib version
        # Set paper-ready matplotlib parameters
        plt.rcParams.update(
            {
                # Use serif fonts - NeurIPS uses Times Roman (ptm)
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                # Font sizes
                "font.size": 14,  # Regular text
                "axes.titlesize": 20,  # Title size
                "axes.labelsize": 14,  # Axis label size
                "xtick.labelsize": 14,  # X tick label size
                "ytick.labelsize": 14,  # Y tick label size
                "legend.fontsize": 12,  # Legend font size
                # Line widths
                "axes.linewidth": 0.5,
                "grid.linewidth": 0.5,
                "lines.linewidth": 1.0,
                "lines.markersize": 3,
                # Clean style for academic publications
                "axes.grid": False,
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "grid.color": "#CCCCCC",
                "grid.linestyle": "--",
                # Legend settings
                "legend.frameon": False,
                "legend.numpoints": 1,
                "legend.handlelength": 2,
                # Use TrueType fonts for better PDF output
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

        max_t = max(
            np.max(ref_times),
            np.max(pred_times),
            np.max(pred_times_rot_45),
            np.max(pred_times_rot_90),
            np.max(pred_times_rot_180),
            np.max(pred_times_translated),
        )
        min_t = min(
            np.min(ref_times),
            np.min(pred_times),
            np.min(pred_times_rot_45),
            np.min(pred_times_rot_90),
            np.min(pred_times_rot_180),
            np.min(pred_times_translated),
        )

        # Create a single figure: 2 rows x 6 columns (times in first row, velocities in second row)
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.3)

        # Row 1: Time plots (6 columns)
        time_plots = [
            (ref_times.T, "Reference Time"),
            (pred_times.T, "Predicted Time"),
            (pred_times_rot_45.T, "Pred. Time Rot. 45°"),
            (pred_times_rot_90.T, "Pred. Time Rot. 90°"),
            (pred_times_rot_180.T, "Pred. Time Rot. 180°"),
            (pred_times_translated.T, "Pred. Time Translated"),
        ]

        for idx, (data, title) in enumerate(time_plots):
            im = axes[0, idx].imshow(
                data,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="jet",
                vmin=min_t,
                vmax=max_t,
            )
            axes[0, idx].plot(
                Xs[num_ns[0] // 2, num_ns[1] // 2][0],
                Xs[num_ns[0] // 2, num_ns[1] // 2][1],
                "r*",
                markersize=20,
            )
            axes[0, idx].set_title(title, pad=10)
            axes[0, idx].set_aspect("equal")
            # axes[0, idx].set_xticks([])
            # axes[0, idx].set_yticks([])
            cbar = fig.colorbar(im, ax=axes[0, idx], shrink=0.66, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=12)

        # Row 2: Velocity plots (6 columns)
        vel_plots = [
            (vel.T, "Original Velocity"),
            (rec_vel.T, "Recon. Velocity"),
            (rec_vel_rot_45.T, "Recon. Vel. Rot. 45°"),
            (rec_vel_rot_90.T, "Recon. Vel. Rot. 90°"),
            (rec_vel_rot_180.T, "Recon. Vel. Rot. 180°"),
            (rec_vel_translated.T, "Recon. Vel. Translated"),
        ]

        for idx, (data, title) in enumerate(vel_plots):
            im = axes[1, idx].imshow(
                data,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, idx].plot(
                Xs[num_ns[0] // 2, num_ns[1] // 2][0],
                Xs[num_ns[0] // 2, num_ns[1] // 2][1],
                "r*",
                markersize=20,
            )
            axes[1, idx].set_title(title, pad=10)
            axes[1, idx].set_aspect("equal")
            # axes[1, idx].set_xticks([])
            # axes[1, idx].set_yticks([])
            cbar = fig.colorbar(im, ax=axes[1, idx], shrink=0.66, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(final_path + f"/{name}_equiv.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def visualize_gt_euclidean_3D(
    grid_data,
    vel,
    pred_vel,
    ref_times,
    pred_times,
    gradients,
    name,
    vmin=0.0,
    vmax=1.0,
    sp=2,
    all_indices=False,
):
    x = grid_data["x"]
    y = grid_data["y"]
    z = grid_data["z"]
    points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

    X, Y, Z = np.transpose(points, (3, 0, 1, 2))
    # pred_vel = pred_vel[::sp, ::sp, ::sp]
    # vel = vel[::sp, ::sp, ::sp]

    num_ns = vel.shape

    for ixs in [
        (1, 1, 1),
        # (num_ns[0] // 2, num_ns[1] // 2, num_ns[2] // 2),
        # (num_ns[0] - 2, num_ns[1] - 2, num_ns[2] - 2),
    ]:
        pred_times_sample = pred_times[ixs]
        ref_times_saple = ref_times[ixs]
        min_t = ref_times_saple.min()
        max_t = ref_times_saple.max()

        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
                [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
            ],
        )

        sample_vel = pred_vel[ixs][..., 1]
        levels = np.linspace(min_t, max_t, 10)
        levels = [levels[1], levels[3], levels[5]]

        for i, l in enumerate(levels):

            fig.add_trace(
                go.Volume(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=sample_vel.flatten(),
                    cmin=vmin,
                    cmax=vmax,
                    colorscale="viridis",
                    opacity=0.1,  # needs to be small to see through all surfaces
                    surface_count=21,  # needs to be a large number for good volume rendering
                ),
                row=1,
                col=i + 1,
            )

            fig.add_trace(
                go.Volume(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=vel.flatten(),
                    cmin=vmin,
                    cmax=vmax,
                    colorscale="viridis",
                    opacity=0.1,  # needs to be small to see through all surfaces
                    surface_count=21,  # needs to be a large number for good volume rendering
                ),
                row=2,
                col=i + 1,
            )

            fig.add_trace(
                go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    name="Pred times",
                    isomin=l,
                    isomax=l,
                    cmin=min_t - 1e-2,
                    cmax=max_t + 1e-2,
                    surface_count=1,
                    colorscale="Portland",
                    value=pred_times_sample.flatten(),
                    opacity=1,
                    # caps=dict(x_show=False, y_show=False, z_show=False),
                ),
                row=1,
                col=i + 1,
            )

            fig.add_trace(
                go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    name="FMM",
                    isomin=l,
                    isomax=l,
                    cmin=min_t - 1e-2,
                    cmax=max_t + 1e-2,
                    colorscale="Portland",
                    surface_count=1,
                    value=ref_times_saple.flatten(),
                    opacity=1,
                    # caps=dict(x_show=False, y_show=False, z_show=False),
                ),
                row=2,
                col=i + 1,
            )

        fig.update(layout_coloraxis_showscale=False)

        # Convert Plotly figure to image in memory
        img_bytes = fig.to_image(format="png", scale=2)  # No need to save to disk
        img = Image.open(io.BytesIO(img_bytes))  # Convert bytes to PIL image
        logger.log_image(f"{name}_{ixs}", img)

        # Close the figure to free up memory
        plt.close(fig)


def visualize_gt_position_orientation(
    grid_data,
    vel,
    pred_vel,
    ref_times,
    pred_times,
    gradients,
    name,
    vmin=0.0,
    vmax=1.0,
    sp=2,
    all_indices=False,
):
    x = grid_data["x"]
    y = grid_data["y"]
    Xs = grid_data["Xs"]
    num_ns = Xs.shape
    gradients = gradients[..., 2:]

    for ixs in [
        (1, 1, 1),
        # (num_ns[0] // 2, num_ns[1] // 2),
        # (num_ns[0] - 2, num_ns[1] - 2),
    ]:

        colors = ["black", "black"]
        tmapref = hv.Image((x, y, np.min(ref_times[ixs], axis=-1).T), label="T_ref")
        tmap = hv.Image((x, y, np.min(pred_times[ixs], axis=-1).T), label="T_pred")

        levels = np.linspace(
            np.min(ref_times[ixs], axis=-1).min(),
            np.min(ref_times[ixs], axis=-1).max(),
            15,
        )

        tctrref = hv.operation.contours(tmapref, levels=levels).opts(
            color=colors[0], cmap=[colors[0]], linestyle="solid", linewidth=2
        )
        tctr = hv.operation.contours(tmap, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="solid", linewidth=2
        )

        max_t = max(np.max(ref_times[ixs]), np.max(pred_times[ixs]))
        min_t = min(np.min(ref_times[ixs]), np.min(pred_times[ixs]))
        tmapref = tmapref.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))
        tmap = tmap.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))

        srcp = hv.Scatter([Xs[ixs][:2]]).opts(marker="*", s=200, c="r")

        fig1 = (
            (tmapref * tctrref * srcp)
            .opts(hv.opts.Image(show_legend=False))
            .opts(title="Reference contours Min")
        )

        fig2 = (
            (tmap * tctr * srcp)
            .opts(hv.opts.Image(show_legend=False))
            .opts(title="Solution contours Min")
        )

        fig = fig1 + fig2

        fig = hv.render(fig, backend="matplotlib")
        logger.log_image(f"{name}_{ixs}", fig)

        for i in range(pred_times[ixs].shape[2]):
            sample_pred_times = pred_times[ixs][..., i]
            sample_ref_times = ref_times[ixs][..., i]

            tmapref = hv.Image((x, y, sample_ref_times.T), label="T_ref")
            tmap = hv.Image((x, y, sample_pred_times.T), label="T_pred")

            levels = np.linspace(
                sample_ref_times.min(),
                sample_ref_times.max(),
                15,
            )

            tctrref = hv.operation.contours(tmapref, levels=levels).opts(
                color=colors[0], cmap=[colors[0]], linestyle="solid", linewidth=2
            )
            tctr = hv.operation.contours(tmap, levels=levels).opts(
                color=colors[1], cmap=[colors[1]], linestyle="solid", linewidth=2
            )

            tmapref = tmapref.opts(
                cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8)
            )
            tmap = tmap.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))

            fig1 = (
                (tmapref * tctrref * srcp)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title=f"Reference contours {i}")
            )

            fig2 = (
                (tmap * tctr * srcp)
                .opts(hv.opts.Image(show_legend=False))
                .opts(title=f"Solution contours {i}")
            )

            fig = fig1 + fig2

            fig = hv.render(fig, backend="matplotlib")
            logger.log_image(f"{name}_{ixs}_{i}", fig)
            plt.close(fig)


def visualize_gt_spherical(
    grid_data,
    vel,
    pred_vel,
    ref_times,
    pred_times,
    gradients,
    name,
    vmin=0.0,
    vmax=1.0,
    sp=2,
    all_indices=False,
    final_path=None,
):

    x = grid_data["x"]
    y = grid_data["y"]
    Xs = grid_data["Xs"]
    num_ns = Xs.shape
    gradients = gradients[..., 2:]

    if all_indices:
        indices = [(i, j) for i in range(num_ns[0]) for j in range(num_ns[1])]
    else:
        indices = [
            (1, 1),
            (num_ns[0] // 2, num_ns[1] // 2),
            (num_ns[0] - 2, num_ns[1] - 2),
        ]

    for ixs in indices:

        vmap = hv.Image(
            (x, y, vel.T),
            kdims=["X", "Y"],
            vdims="Velocity",
            label="V, ",
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        sample_vel = pred_vel[ixs][..., 1]

        rec_vmap = hv.Image(
            (x, y, sample_vel.T), kdims=["X", "Y"], vdims="Velocity", label="V, "
        ).opts(cmap="viridis", colorbar=True, clim=(vmin, vmax))

        colors = ["black", "white"]
        max_t = max(np.max(ref_times[ixs]), np.max(pred_times[ixs]))
        min_t = min(np.min(ref_times[ixs]), np.min(pred_times[ixs]))
        tmapref = hv.Image((x, y, ref_times[ixs].T), label="T_ref")
        tmap = hv.Image((x, y, pred_times[ixs].T), label="T_pred")

        levels = np.linspace(ref_times[ixs].min(), ref_times[ixs].max(), 15)

        tctrref = hv.operation.contours(tmapref, levels=levels).opts(
            color=colors[0], cmap=[colors[0]], linestyle="solid", linewidth=4
        )
        tctr = hv.operation.contours(tmap, levels=levels).opts(
            color=colors[1], cmap=[colors[1]], linestyle="dashed", linewidth=2
        )

        srcp = hv.Scatter([Xs[ixs]]).opts(marker="*", s=200, c="r")

        gradients_ = gradients[ixs][::sp, ::sp]
        mag = np.linalg.norm(gradients_, axis=-1) * 0.05
        angle = (np.pi / 2.0) - np.arctan2(
            gradients_[..., 0] / mag, gradients_[..., 1] / mag
        )
        # angle = -np.arctan2(gradients_[..., 1]/mag, gradients_[..., 0]/mag)
        vf = (
            hv.VectorField((x[::sp], y[::sp], angle.T, mag.T))
            .opts(magnitude="Magnitude")
            .opts(hv.opts.VectorField(pivot="tail"))
        )

        fig1 = (
            (vmap * tctrref * tctr * srcp)
            .opts(hv.opts.Image(show_legend=False))
            .opts(title="Solution contours")
        )

        fig2 = (
            (vmap * srcp * vf)
            .opts(hv.opts.Image(show_legend=False))
            .opts(title="Gradients")
        )

        tmapref = tmapref.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))
        tmap = tmap.opts(cmap="jet", colorbar=True, clim=(min_t, max_t + 1e-8))

        aux_data = np.abs(ref_times[ixs].T - pred_times[ixs].T)

        tmapdiff = hv.Image((x, y, aux_data), label="T_ref").opts(
            cmap="jet",
            colorbar=True,
            clim=(np.min(aux_data), np.max(aux_data) + 1e-8),
        )

        fig3 = (
            (tmapref * srcp)
            .opts(hv.opts.Image(show_legend=False))
            .opts(title="Ground Truth Time")
        )

        fig4 = (
            (tmap * srcp).opts(hv.opts.Image(show_legend=False)).opts(title="Pred Time")
        )

        fig5 = (tmapdiff).opts(hv.opts.Image(show_legend=False)).opts(title="Diff Time")

        fig = (
            (rec_vmap * srcp).opts(title="Recon vel") + fig1 + fig2 + fig3 + fig4 + fig5
        ).cols(3)
        fig = hv.render(fig, backend="matplotlib")

        # MODIFIED: Use logger.log_image instead of direct wandb call
        logger.log_image(f"{name}_{ixs}", fig)
        plt.close(fig)
