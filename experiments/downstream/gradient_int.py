import numpy as np
import torch
from time import time
import jax.numpy as jnp
import jax

import warnings
from matplotlib import pyplot as plt

import holoviews as hv  # for visualization

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

hv.extension("matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def _all_coords_no_plot(solver, solver_param, latents, num_pairs_batch_test):
    """Process all coordinates with plotting requirements (including gradients and velocity)"""
    # Only process times (no plotting)

    p, a = latents

    theta_range = [-jnp.pi, jnp.pi]

    xmin, ymin, thetamin = [-4.0, -4.0] + [theta_range[0]]

    xmax, ymax, thetamax = [4.0, 4.0] + [theta_range[1]]

    x = np.linspace(xmin, xmax, 100, dtype=np.float32)
    y = np.linspace(ymin, ymax, 100, dtype=np.float32)
    theta = np.linspace(thetamin, thetamax, 100, endpoint=False, dtype=np.float32)
    xr = x
    yr = y
    thetar = theta

    Xr = np.stack(np.meshgrid(xr, yr, thetar, indexing="ij"), axis=-1)

    xs = np.array([0.0])
    ys = np.array([0.0])
    thetas = np.array([0.0])

    Xs = np.stack(np.meshgrid(xs, ys, thetas, indexing="ij"), axis=-1)

    coords = np.stack(
        np.meshgrid(xs, ys, thetas, xr, yr, thetar, indexing="ij"), axis=-1
    ).reshape(-1, 2, 3)
    coords = coords[None, ...]

    apply_solver_times_jitted = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.times)
    )

    num_pairs_per_vel = coords.shape[1]

    # Calculate number of full batches and remainder information
    num_full_batches = num_pairs_per_vel // num_pairs_batch_test
    remainder_start = num_full_batches * num_pairs_batch_test
    has_remainder = remainder_start < num_pairs_per_vel

    # Initialize accumulators for gradients and velocity
    pred_times_all = jnp.zeros((coords.shape[0], 0), dtype=coords.dtype)

    # Process full batches
    for batch_idx in range(num_full_batches):
        start_idx = batch_idx * num_pairs_batch_test
        subcoords = jax.lax.dynamic_slice(
            coords,
            (0, start_idx, 0, 0),  # start indices for all 4 dimensions
            (
                coords.shape[0],
                num_pairs_batch_test,
                coords.shape[2],
                coords.shape[3],
            ),  # slice sizes for all 4 dimensions
        )

        # Process batch
        pred_times = jax.lax.stop_gradient(
            apply_solver_times_jitted(
                solver_param,
                inputs=subcoords,
                p=p,
                a=a,
            )
        )

        pred_times = jnp.asarray(pred_times, dtype=subcoords.dtype)

        # Concatenate results
        if batch_idx == 0:
            pred_times_all = pred_times
        else:
            pred_times_all = jnp.concatenate((pred_times_all, pred_times), axis=1)

    # Process remainder if it exists
    if has_remainder:
        remainder_size = num_pairs_per_vel - remainder_start
        remainder = jax.lax.dynamic_slice(
            coords,
            (0, remainder_start, 0, 0),  # start indices for all 4 dimensions
            (
                coords.shape[0],
                remainder_size,
                coords.shape[2],
                coords.shape[3],
            ),
        )
        pred_remainder = jax.lax.stop_gradient(
            apply_solver_times_jitted(
                solver_param,
                inputs=remainder,
                p=p,
                a=a,
            )
        )

        pred_remainder = jnp.asarray(pred_remainder, dtype=remainder.dtype)

        if num_full_batches > 0:
            pred_times_all = jnp.concatenate((pred_times_all, pred_remainder), axis=1)
        else:
            pred_times_all = pred_remainder

        time = jnp.min(pred_times_all.reshape(100, 100, 100), axis=-1)

        # plot isocurves and save

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # Create a grid of x, y points
        X, Y = np.meshgrid(x, y)

        # Create the plot
        fig = plt.figure(figsize=(10, 8))

        # Plot the contour lines with labels
        contour_lines = plt.contour(X, Y, time.T, 10, colors="black", linewidths=0.5)

        # Add labels and title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save the figure to a file
        plt.savefig("isocurve_plot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def shortest_path(
    start,
    goal,
    solver,
    params,
    latents,
    step_size=0.01,
    epsilon=0.05,
    max_iter=500,
    bidirectional=False,
):
    # Convert inputs to JAX arrays
    if not isinstance(start, jnp.ndarray):
        start = jnp.array(start)

    if not isinstance(goal, jnp.ndarray):
        goal = jnp.array(goal)

    p, a = latents

    # Stack the start and goal points
    xs_xr = jnp.expand_dims(jnp.expand_dims(jnp.vstack([start, goal]), 0), 0)

    # Initial distance calculation
    apply_distance = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.distance)
    )

    apply_project = jax.jit(jax.tree_util.Partial(solver.apply, method=solver.project))

    xs_xr = apply_project(params, xs_xr)

    apply_solver_times_grad_vel_jitted = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.times_grad_vel, aux_vel=False)
    )

    # Define a single optimization step as a jitted function
    @jax.jit
    def optimization_step(xs_xr):
        # Get gradients and velocities
        _, gradients, vel = apply_solver_times_grad_vel_jitted(params, xs_xr, p, a)

        # Update points
        # norm = jnp.sqrt(jnp.sum(gradients**2, axis=-1, keepdims=True))

        if not bidirectional:
            mask = jnp.zeros_like(gradients)
            mask = mask.at[0, 0, 1].set(1.0)
            gradients = mask * gradients

        # Apply the masked update
        xs_xr = xs_xr - step_size * gradients * (vel[..., None] ** 2)

        xs_xr = apply_project(params, xs_xr)

        # Recalculate distance
        dis = apply_distance(params, xs_xr)

        return xs_xr, dis

    _ = optimization_step(xs_xr)

    dis = apply_distance(params, xs_xr)
    print(dis, xs_xr)
    start_time = time()

    # Initialize lists to store path points
    point0 = [np.array(xs_xr[0, 0, 0])]
    point1 = [np.array(xs_xr[0, 0, 1])]

    # Main optimization loop
    iter_count = 0
    while dis > epsilon:
        print("iter", iter_count, dis)
        # Perform optimization step
        xs_xr, dis = optimization_step(xs_xr)

        # Store path points (must be outside jit since it has side effects)
        point1.append(np.array(xs_xr[0, 0, 1]))
        if bidirectional:
            point0.append(np.array(xs_xr[0, 0, 0]))

        iter_count += 1

    end_time = time()
    print("time", end_time - start_time)

    # Reverse point1 and concatenate with point0 to form the complete path
    point1.reverse()
    point = point0 + point1

    # Convert to numpy array
    res = np.stack(point, axis=0)

    return res


def plot_curve_euclidean_2d(points, vel, x_min=None, x_max=None, vmin=0.0, vmax=1.0):
    points = [tuple(row) for row in points.tolist()]

    start = points[0]
    goal = points[-1]

    nx, ny = vel.shape
    xmin, ymin = x_min if x_min is not None else [-1.0, -1.0]
    xmax, ymax = x_max if x_max is not None else [1.0, 1.0]
    print(xmax, ymax, xmin, ymin)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    figs = []

    vmap = (
        hv.Image((x, y, vel.T), kdims=["X", "Y"], vdims="Velocity")
        .opts(cmap="gray", colorbar=True, clim=(vmin, vmax), alpha=0.8)
        .opts(fig_size=200)
    )

    start_plot = (
        hv.Scatter(start, label="Start")
        .opts(marker="*", s=500, c="tab:red", edgecolor="black")
        .opts(fig_size=200)
    )
    goal_plot = (
        hv.Scatter(goal, label="Goal")
        .opts(marker="s", s=200, c="tab:red", edgecolor="black", linewidth=2)
        .opts(fig_size=200)
    )

    path = hv.Path([points], label="Path")
    path.opts(color="tab:green", linewidth=8, show_legend=True)
    overlay = vmap * path * start_plot * goal_plot
    overlay.opts(legend_position="top")
    overlay.opts(fontscale=1.5)
    return overlay


if __name__ == "__main__":
    from experiments.downstream.utils.autodecoding_import import autodecoding_import

    (
        solver,
        solver_params,
        autodecoder,
        autodecoder_params,
        loader,
        (vmin, vmax),
        (x_min, x_max),
    ) = autodecoding_import(
        "n7z7ahnv",
        # "qxg8hewu",
        # "qkad0tb2",
        # "s6d9891u",
        # "oo6ofzl8",
        # "gezb2l8x",
        # "g56vmljo",
        # "0to0nonx",
        # "a8zobuqr",
        # "svm3fmqc",
        aux_out=True,
        train=False,
        num_epochs_auto=1,
    )

    # solver.epsilon = 0.5
    # solver.xi = 1

    idx_vel = 8
    # latents = autodecoder.apply(autodecoder_params, jnp.asarray([idx_vel]))
    # save the latents
    # np.save("latents_0.npy", np.array(latents[0][0]), allow_pickle=True)
    # np.save("latents_1.npy", np.array(latents[0][1]), allow_pickle=True)
    # np.save("latents_2.npy", np.array(latents[1]), allow_pickle=True)
    latents = (
        (
            jnp.asarray(np.load("latents_0.npy", allow_pickle=True)),
            jnp.asarray(np.load("latents_1.npy", allow_pickle=True)),
        ),
        jnp.asarray(np.load("latents_2.npy", allow_pickle=True)),
    )

    # _all_coords_no_plot(
    #     solver=solver,
    #     solver_param=solver_params,
    #     latents=latents,
    #     num_pairs_batch_test=2 * 5120,
    # )

    # for i in range(len(loader.dataset.base_dataset)):
    #     vel = loader.dataset.base_dataset[i][0]
    #     vel = jnp.min(vel, axis=-1)
    #     plt.imshow(vel.T, cmap="gray")
    #     plt.savefig(f"vel_{i}.png", dpi=300, bbox_inches="tight")
    #     plt.close()

    vel = loader.dataset.base_dataset[idx_vel][0]

    points = shortest_path(
        start=[3, -3, 90 * np.pi / 180],
        goal=[-3, 0.5, 0 * np.pi / 180],
        solver=solver,
        params=solver_params,
        latents=latents,
        step_size=1e-3,
        epsilon=0.5,
        max_iter=1e8,
    )

    angles = points[:, 2]
    points = points[:, 0:2]
    # plot angles
    plt.plot(angles)
    plt.savefig("angles.png")
    plt.close()

    vel = jnp.min(vel, axis=-1)

    # plot all the velocities and save them

    print(points.shape)

    overlay = plot_curve_euclidean_2d(
        points, vel, x_min=x_min, x_max=x_max, vmin=vmin, vmax=vmax
    )

    hv.save(overlay, "test", fmt="png")
