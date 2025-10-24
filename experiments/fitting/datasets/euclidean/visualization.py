import holoviews as hv  # for visualization

hv.extension("matplotlib")
import numpy as np


def plot_single_velocity(velocity, x_min, x_max, path, vmin=None, vmax=None):
    if not isinstance(velocity, np.ndarray):
        velocity = velocity.numpy()

    if vmin is None or vmax is None:
        vmax, vmin = np.max(velocity), np.min(velocity)

    nx, ny = velocity.shape
    xmin, ymin = x_min
    xmax, ymax = x_max
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    plot = hv.Image((x, y, velocity.T), kdims=["X", "Y"], vdims="Velocity").opts(
        cmap="viridis", colorbar=True, alpha=0.8, clim=(vmin, vmax)
    )

    hv.save(plot, path)


def plot_interpolated_values(
    velocity, coords, interpolated_values, x_min, x_max, path=None, v_max=1
):
    """
    Plots the 2D image with sampled coordinates and their corresponding interpolated values.

    Args:
    - image (torch.Tensor): The original 2D image (H, W) or (C, H, W).
    - coords (torch.Tensor): Coordinates used for sampling (num_points, 2), normalized [-1, 1].
    - interpolated_values (torch.Tensor): Interpolated values at the sampled coordinates (num_points, channels).
    """
    # Convert normalized coordinates [x_min, xmax] to pixel coordinates

    if not isinstance(velocity, np.ndarray):
        velocity = velocity.numpy()
    coords = coords.view(-1, 2).cpu().numpy()
    interpolated_values = interpolated_values.view(-1).cpu().numpy()

    if vmin is None or vmax is None:
        vmax, vmin = np.max(velocity), np.min(velocity)

    nx, ny = velocity.shape
    xmin, ymin = x_min
    xmax, ymax = x_max
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    vmap = hv.Image((x, y, velocity.T), kdims=["X", "Y"], vdims="Velocity").opts(
        cmap="viridis", colorbar=True, clim=(vmin, vmax), alpha=0.8
    )

    # Create the points
    points = hv.Points(
        (coords[:, 0], coords[:, 1], interpolated_values),
        kdims=["X", "Y"],
        vdims="Value",
    ).opts(
        s=50, cmap="viridis", color=hv.dim("Value"), clim=(vmin, vmax), edgecolor="red"
    )

    plot = vmap * points

    if path is not None:
        hv.save(plot, path)

    return plot
