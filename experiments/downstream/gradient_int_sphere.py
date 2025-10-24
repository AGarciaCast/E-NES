import numpy as np
from time import time
import jax.numpy as jnp
import jax


import warnings


jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


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

    apply_ambient2chart = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.ambient2chart)
    )

    apply_jac_fn_transform = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.jac_fn_transform)
    )

    # jacobian = apply_jac_fn_transform(xs_xr)
    # jacobian = apply_jac_fn_transform(xs_xr)
    # print("jacobian", jacobian.shape)

    apply_fn_transform = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.fn_transform)
    )

    apply_solver_times_grad_vel_jitted = jax.jit(
        jax.tree_util.Partial(solver.apply, method=solver.times_grad_vel, aux_vel=False)
    )

    # Define a single optimization step as a jitted function
    # @jax.jit
    def optimization_step(xs_xr):
        # Get gradients and velocities
        _, gradients, vel = apply_solver_times_grad_vel_jitted(params, xs_xr, p, a)

        # Update points
        # norm = jnp.sqrt(jnp.sum(gradients**2, axis=-1, keepdims=True))

        if not bidirectional:
            mask = jnp.zeros_like(gradients)
            mask = mask.at[0, 0, 1].set(1.0)
            gradients = mask * gradients

        # Normalize gradients
        gradients = gradients * (vel[..., None] ** 2)

        jacobian = apply_jac_fn_transform(params, xs_xr)
        gradients = jnp.einsum("bspij,bspj->bspi", jacobian, gradients)

        # Apply the masked update
        xs_xr = apply_fn_transform(params, xs_xr) - step_size * gradients

        xs_xr = apply_ambient2chart(params, xs_xr)

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


if __name__ == "__main__":
    import os
    import pickle
    from experiments.downstream.utils.autodecoding_import import autodecoding_import

    solver = None
    # Configuration
    idx_vel = 56
    start = [1.5, 1]
    goal = [4, 2]
    force_recompute = False  # Set to True to force recomputation

    # wandb_id = "0e1cqt34"
    wandb_id = "y10qqb2n"
    for idx_vel in range(0, 100):

        # Cache file paths
        cache_dir = "/experiments/downstream/results"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"path_{wandb_id}_vel_{idx_vel}_start_{start[0]}_{start[1]}_goal_{goal[0]}_{goal[1]}.pkl",
        )

        # Try to load from cache
        if os.path.exists(cache_file) and not force_recompute:
            print(f"Loading cached data from {cache_file}...")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                points = cached_data["points"]
                vel = cached_data["vel"]
                vmin = cached_data["vmin"]
                vmax = cached_data["vmax"]
                x_min = cached_data["x_min"]
                x_max = cached_data["x_max"]
            print(f"Loaded cached data. Points shape: {points.shape}")
        else:
            print("Computing path and loading velocity field...")
            if solver is None:
                (
                    solver,
                    solver_params,
                    autodecoder,
                    autodecoder_params,
                    loader,
                    (vmin, vmax),
                    (x_min, x_max),
                ) = autodecoding_import(
                    wandb_id,
                    aux_out=True,
                    train=False,
                    num_epochs_auto=100,
                )

            latents = autodecoder.apply(autodecoder_params, jnp.asarray([idx_vel]))
            vel = loader.dataset.base_dataset[idx_vel][0]

            points = shortest_path(
                start=start,
                goal=goal,
                solver=solver,
                params=solver_params,
                latents=latents,
                step_size=5e-3,
                epsilon=0.1,
                max_iter=1e8,
            )

            # Save to cache
            print(f"Saving to cache: {cache_file}")
            cache_data = {
                "points": points,
                "vel": vel,
                "vmin": vmin,
                "vmax": vmax,
                "x_min": x_min,
                "x_max": x_max,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Cached data saved. Points shape: {points.shape}")

        print(points.shape)
