from experiments.fitting.utils.logging import logger  # Add this line
import warnings
import numpy as np
import jax.numpy as jnp
import holoviews as hv  # for visualization

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image
import jax


hv.extension("matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib.pyplot as plt


def visualize_reconstructions_euclidean_2D(
    targets,
    coords,
    solver_fn,
    solver_params,
    p,
    a,
    max_num_visualized_rec,
    max_pairs_plot,
    rng,
    vmin=0.0,
    vmax=1.0,
    x_min=None,
    x_max=None,
    name="recon",
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
    indices = jnp.arange(coords.shape[1])
    mask = jax.random.permutation(rng, indices)[:max_pairs_plot]
    p_pos, p_ori = p
    coords = jnp.asarray(coords[:, mask], dtype=p_pos.dtype)

    ############################################################################################################
    # Rotate poses
    ############################################################################################################
    theta = np.pi / 4

    R = jnp.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
        dtype=p_pos.dtype,
    )

    # JAX arrays are immutable, so we create new arrays instead of cloning
    p_pos_rotated = jnp.einsum("ij,bnj->bni", R, p_pos)  # Rotate the position

    if p_ori is not None:
        p_ori_rotated = jnp.einsum("ij,bnjk->bnik", R, p_ori)
        p_rotated = (p_pos_rotated, p_ori_rotated)
    else:
        p_rotated = (p_pos_rotated, None)

    ############################################################################################################
    # Translate poses
    ############################################################################################################
    translation = jnp.asarray(0.5, dtype=p_pos.dtype)

    p_pos_translated = p_pos + translation
    p_translated = (p_pos_translated, p_ori)

    ############################################################################################################
    # Flip poses
    ############################################################################################################

    # Create a new array with flipped y coordinates
    p_pos_flip = p_pos.at[..., 1].multiply(-1.0)  # Flip the position

    if p_ori is not None:
        p_ori_flip = p_ori.at[..., 1].multiply(-1.0)
        p_flip = (p_pos_flip, p_ori_flip)
    else:
        p_flip = (p_pos_flip, None)

    # Get the outputs for the batch, and its rotated and translated versions
    outputs = np.array(
        solver_fn(solver_params, inputs=coords, p=p, a=a), dtype=targets[0].dtype
    )
    outputs_rot = np.array(
        solver_fn(solver_params, inputs=coords, p=p_rotated, a=a),
        dtype=targets[0].dtype,
    )
    outputs_translated = np.array(
        solver_fn(solver_params, inputs=coords, p=p_translated, a=a),
        dtype=targets[0].dtype,
    )
    outputs_flip = np.array(
        solver_fn(solver_params, inputs=coords, p=p_flip, a=a), dtype=targets[0].dtype
    )

    nx, ny = targets[0].shape
    xmin, ymin = x_min if x_min is not None else [-1.0, -1.0]
    xmax, ymax = x_max if x_max is not None else [1.0, 1.0]
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    # Plot the first n images
    for i, rec_vel in enumerate(outputs):
        if i > max_num_visualized_rec:
            break

        true_vel = targets[i]

        # Mask the coordinates and labels
        sample_coords = np.array(coords[i].reshape(-1, 2))
        sample_p_pos = np.array(p_pos[i])
        rec_vel = np.array(rec_vel.reshape(-1))
        rec_vel_rot = np.array(outputs_rot[i].reshape(-1))
        rec_vel_translated = np.array(outputs_translated[i].reshape(-1))
        rec_vel_flip = np.array(outputs_flip[i].reshape(-1))

        vmap = hv.Image((x, y, true_vel.T), kdims=["X", "Y"], vdims="Velocity").opts(
            cmap="viridis", colorbar=True, clim=(vmin, vmax), alpha=0.8
        )

        if p_ori is not None:
            sample_p_ori = np.array(p_ori[i])

            if len(sample_p_ori.shape) == 3:
                sample_p_ori = sample_p_ori[..., 0]

            # Create the point plot
            latents = hv.Points(
                (sample_p_pos[:, 0], sample_p_pos[:, 1]), kdims=["X", "Y"]
            ).opts(s=25, c="red")

            mag = (
                np.linalg.norm(sample_p_ori, axis=-1) * 0.01
            )  # Scale magnitude appropriately
            angle = np.arctan2(
                sample_p_ori[..., 1], sample_p_ori[..., 0]
            )  # Correct use of arctan2

            # Flatten arrays
            x_flat = sample_p_pos[:, 0].flatten()
            y_flat = sample_p_pos[:, 1].flatten()
            angle_flat = angle.flatten()
            mag_flat = mag.flatten()

            # Create a VectorField in HoloViews
            vector_data = np.column_stack([x_flat, y_flat, angle_flat, mag_flat])
            vector_field = hv.VectorField(vector_data).opts(
                color="red",
                magnitude="Magnitude",
                pivot="tail",
                aspect="equal",
            )

            latents = latents * vector_field

        else:
            latents = hv.Points(
                (sample_p_pos[:, 0], sample_p_pos[:, 1]), kdims=["X", "Y"]
            ).opts(s=25, c="red")

        # Create the points

        points_vel = hv.Points(
            (sample_coords[:, 0], sample_coords[:, 1], rec_vel),
            kdims=["X", "Y"],
            vdims="Value",
        ).opts(
            s=50,
            cmap="viridis",
            color=hv.dim("Value"),
            clim=(vmin, vmax),
            edgecolor="red",
        )

        points_vel_rot = hv.Points(
            (sample_coords[:, 0], sample_coords[:, 1], rec_vel_rot),
            kdims=["X", "Y"],
            vdims="Value",
        ).opts(
            s=50,
            cmap="viridis",
            color=hv.dim("Value"),
            clim=(vmin, vmax),
            edgecolor="red",
        )

        points_vel_translated = hv.Points(
            (sample_coords[:, 0], sample_coords[:, 1], rec_vel_translated),
            kdims=["X", "Y"],
            vdims="Value",
        ).opts(
            s=50,
            cmap="viridis",
            color=hv.dim("Value"),
            clim=(vmin, vmax),
            edgecolor="red",
        )

        points_vel_flip = hv.Points(
            (sample_coords[:, 0], sample_coords[:, 1], rec_vel_flip),
            kdims=["X", "Y"],
            vdims="Value",
        ).opts(
            s=50,
            cmap="viridis",
            color=hv.dim("Value"),
            clim=(vmin, vmax),
            edgecolor="red",
        )

        fig = (
            (vmap * points_vel).opts(title="Original Recon")
            + (vmap * latents).opts(title="Original latents")
            + points_vel_rot.opts(title="Rotated Recon")
            + points_vel_translated.opts(title="Translated Recon")
            + points_vel_flip.opts(title="Fliped Recon")
        )

        # fig =  (vmap*latents).opts(title='Original latents')
        fig = hv.render(fig, backend="matplotlib")
        logger.log_image(f"{name}_{i}", fig)

        plt.close(fig)  # Close the figure to free memory


def visualize_reconstructions_spherical(
    targets,
    coords,
    solver_fn,
    solver_params,
    p,
    a,
    max_num_visualized_rec,
    max_pairs_plot,
    rng,
    vmin=0.0,
    vmax=1.0,
    x_min=None,
    x_max=None,
    name="recon",
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
    indices = jnp.arange(coords.shape[1])
    mask = jax.random.permutation(rng, indices)[:max_pairs_plot]
    p_pos, p_ori = p
    coords = jnp.asarray(coords[:, mask], dtype=p_pos.dtype)

    # Get the outputs for the batch, and its rotated and translated versions
    outputs = np.array(
        solver_fn(solver_params, inputs=coords, p=p, a=a), dtype=targets[0].dtype
    )

    nx, ny = targets[0].shape
    xmin, ymin = x_min if x_min is not None else [0.0, 0 * np.pi / 180]
    xmax, ymax = x_max if x_max is not None else [2 * np.pi, np.pi - 0 * np.pi / 180]

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    # Plot the first n images
    for i, rec_vel in enumerate(outputs):
        if i > max_num_visualized_rec:
            break

        true_vel = targets[i]

        # Mask the coordinates and labels
        sample_coords = np.array(coords[i].reshape(-1, 2))
        sample_p_pos = np.array(p_pos[i])
        rec_vel = np.array(rec_vel.reshape(-1))

        vmap = hv.Image((x, y, true_vel.T), kdims=["X", "Y"], vdims="Velocity").opts(
            cmap="viridis", colorbar=True, clim=(vmin, vmax), alpha=0.8
        )

        # Create the points

        points_vel = hv.Points(
            (sample_coords[:, 0], sample_coords[:, 1], rec_vel),
            kdims=["X", "Y"],
            vdims="Value",
        ).opts(
            s=50,
            cmap="viridis",
            color=hv.dim("Value"),
            clim=(vmin, vmax),
            edgecolor="red",
        )

        fig = (vmap * points_vel).opts(title="Original Recon")

        # fig =  (vmap*latents).opts(title='Original latents')
        fig = hv.render(fig, backend="matplotlib")
        logger.log_image(f"{name}_{i}", fig)

        plt.close(fig)  # Close the figure to free memory


def visualize_reconstructions_euclidean_3D(
    targets,
    coords,
    solver_fn,
    solver_params,
    p,
    a,
    max_num_visualized_rec,
    max_pairs_plot,
    rng,
    vmin=0.0,
    vmax=1.0,
    x_min=None,
    x_max=None,
    name="recon",
    label_z="z",
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

    indices = jnp.arange(coords.shape[1])
    mask = jax.random.permutation(rng, indices)[:max_pairs_plot]

    coords = coords[:, mask]

    # Get the outputs for the batch, and its rotated and translated versions
    outputs = np.array(solver_fn(solver_params, inputs=coords, p=p, a=a))

    nx, ny, nz = targets[0].shape
    xmin, ymin, zmin = x_min if x_min is not None else [-1.0, -1.0, 0.0]
    xmax, ymax, zmax = x_max if x_max is not None else [1.0, 1.0, 2.0]
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    X, Y, Z = np.transpose(points, (3, 0, 1, 2))

    # Plot the first n images
    for i, rec_vel in enumerate(outputs):
        if i > max_num_visualized_rec:
            break

        true_vel = targets[i]

        # Mask the coordinates and labels
        sample_coords = np.array(coords[i].reshape(-1, 3))
        rec_vel = np.array(rec_vel.reshape(-1))

        fig = go.Figure(
            data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=true_vel.flatten(),
                cmin=vmin,
                cmax=vmax,
                colorscale="viridis",
                opacity=0.1,  # needs to be small to see through all surfaces
                surface_count=21,  # needs to be a large number for good volume rendering
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=sample_coords[..., 0],
                y=sample_coords[..., 1],
                z=sample_coords[..., 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=rec_vel,  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    cmin=vmin,
                    cmax=vmax,
                    opacity=1,
                ),
            )
        )

        fig = fig.update_layout(
            scene=dict(
                xaxis=dict(title=dict(text="x")),
                yaxis=dict(title=dict(text="y")),
                zaxis=dict(title=dict(text=label_z)),
            ),
            scene_aspectmode="cube",
        )

        # Convert Plotly figure to image in memory
        img_bytes = fig.to_image(format="png", scale=2)  # No need to save to disk
        img = Image.open(io.BytesIO(img_bytes))  # Convert bytes to PIL image
        logger.log_image(f"{name}_{i}", img)
        plt.close(fig)
