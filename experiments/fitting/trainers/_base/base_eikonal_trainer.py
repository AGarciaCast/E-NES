from experiments.fitting.utils.logging import logger  # Add this line
from typing import Any


# For trainstate
from flax import struct, core
import optax
import jax.numpy as jnp


import jax
import time

EPSILON = 1e-8
from experiments.fitting.trainers._base.base_jax_trainer import JaxTrainer

from equiv_eikonal.utils import safe_power, safe_reciprocal


class BaseEikonalTrainer(JaxTrainer):
    """AutoDecoding ENF trainer.

    This trainer is used to train the AutoDecoding ENF model. Some differences with the base ENFTrainer are:
    - We have an additional autodecoder that is trained alongside the enf model.
    - We have a train and val autodecoder.
    - During validation, we fit a new autodecoder for the test images.

    Some similarities with the base enf Trainer are:
    - The training and validation loops are the same, we still have a train_step and val_step.

    Inheriting classes should implement:
    - create_functions: This method should create the training functions.
    - visualize_batch: This method should visualize a batch of data.
    """

    class TrainState(struct.PyTreeNode):
        params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
        solver_opt_state: optax.OptState = struct.field(pytree_node=True)
        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(
        self,
        config,
        solver,
        train_loader,
        val_loader,
        test_loader,
        seed,
        num_epochs,
        visualize_reconstruction=None,
        gt_dataset_val=None,
        gt_dataset_test=None,
        visualize_gt=None,
        visualize_equiv=None,
    ):
        super().__init__(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            seed=seed,
            num_epochs=num_epochs,
        )

        # Store the solver and the optimizer
        self.solver = solver
        self.solver_opt = None

        self.epsilon = EPSILON
        self.power = self.config.eikonal.power
        self.hamiltonian = self.config.eikonal.hamiltonian
        # self.abs_loss = self.config.eikonal.absolute

        self.visualize_reconstruction = visualize_reconstruction
        self.gt_dataset_val = gt_dataset_val
        self.gt_dataset_test = gt_dataset_test
        self.visualize_gt = visualize_gt
        self.visualize_equiv = visualize_equiv

        # Set the number of images to log
        self.num_logged_samples = min(
            [config.logging.num_logged_samples, len(train_loader)]
        )

        self.num_pairs_batch_test = (
            self.config.data.test_batch_size * self.config.data.num_pairs
        )

    def create_functions(self):
        """Create training functions."""
        self.apply_solver_times_grad_vel_jitted = jax.jit(
            jax.tree_util.Partial(
                self.solver.apply, method=self.solver.times_grad_vel, aux_vel=False
            )
        )

        self.apply_solver_times_jitted = jax.jit(
            jax.tree_util.Partial(self.solver.apply, method=self.solver.times)
        )

        self.apply_solver_velocities_jitted = jax.jit(
            jax.tree_util.Partial(self.solver.apply, method=self.solver.velocities)
        )

    def eiko_loss(self, norm_input_grad, vel, reduce_batch=True, abs_loss=False):

        if self.hamiltonian:
            lhs = norm_input_grad * vel
            rhs = jnp.ones_like(vel)
        else:
            lhs = norm_input_grad
            rhs = safe_reciprocal(vel)

        eikp = safe_power(lhs, self.power) - safe_power(rhs, self.power)
        res = eikp / self.power

        if abs_loss:
            res = jnp.abs(res)
        else:
            res = jnp.logaddexp(res, -res) - jnp.log(2.0).astype(res.dtype)

        # res = jnp.abs(res)

        if reduce_batch:
            return jnp.sum(res) / jnp.asarray(res.shape[0] * res.shape[1], jnp.float32)
        else:
            return jnp.sum(res, axis=(1, 2)) / jnp.asarray(res.shape[1], jnp.float32)

    def step(self, state, batch, train=True):
        """Implements a single training/validation step."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def train_epoch(self, state):
        """Trains the model for one epoch.

        Args:
            state: The current training state.
        Returns:
            state: The updated training state.
        """
        # Loop over batches
        losses = 0
        mse_tot = 0
        total_train_points = 0

        # Store current global step at beginning of epoch for consistent logging
        epoch_start_step = self.global_step

        # Group metrics to reduce wandb API calls
        log_frequency = self.config.logging.log_every_n_steps
        commit_frequency = max(
            5, log_frequency * 5
        )  # Commit less frequently than logging

        for batch_idx, batch in enumerate(self.train_loader):

            loss, mse, states = self.train_step(state, batch)

            # After the jitted call, force synchronization
            # This won't affect JIT compilation because it's outside the jitted function
            if type(states) is tuple:
                state, inner_state = states
            else:
                state = states
                inner_state = states

            # Force values to be computed
            loss_value = float(jax.device_get(loss))
            mse_value = float(jax.device_get(mse))

            num_points_batch = batch[0].shape[0] * batch[0].shape[1] * 2.0
            total_train_points += num_points_batch

            losses += loss_value
            mse_tot += mse_value
            # Increment global step
            self.global_step += 1

            # Log every n steps - but only commit occasionally to reduce API calls
            if batch_idx % log_frequency == 0:
                # Add to buffer without committing yet
                logger.add_to_buffer(
                    {
                        "train_eiko_step": loss_value,
                        "train_mse_step": mse_value / num_points_batch,
                    }
                )

                # Only commit every X batches or on the last batch
                should_commit = (batch_idx % commit_frequency == 0) or (
                    batch_idx == len(self.train_loader) - 1
                )
                if should_commit:
                    logger.flush_buffer(step=self.global_step)

                self.update_prog_bar(step=batch_idx)

            if (
                self.visualize_reconstruction is not None
                and not self.config.logging.debug
                and self.global_step % self.config.logging.visualize_every_n_steps == 0
            ):
                inner_state = self.visualize_batch(
                    inner_state,
                    batch,
                    base_dataset=self.train_loader.dataset.base_dataset,
                    name="train/recon",
                )

        # Update epoch loss - log but don't commit yet to batch with validation results
        self.metrics["train_eiko_epoch"] = losses / len(self.train_loader)
        self.metrics["train_mse_epoch"] = mse_tot / total_train_points

        logger.add_to_buffer(
            {
                "train_eiko_epoch": self.metrics["train_eiko_epoch"],
                "train_mse_epoch": self.metrics["train_mse_epoch"],
            },
            commit=False,
        )

        return state

    def _all_coords_for_plot(self, solver_param, coords, p, a):
        """Process all coordinates without plotting requirements"""
        num_pairs_per_vel = coords.shape[1]

        # Calculate number of full batches and remainder information
        num_full_batches = num_pairs_per_vel // self.num_pairs_batch_test
        remainder_start = num_full_batches * self.num_pairs_batch_test
        has_remainder = remainder_start < num_pairs_per_vel

        # Initialize accumulators for gradients and velocity
        pred_times_all = jnp.zeros((coords.shape[0], 0), dtype=coords.dtype)
        pred_vel_all = jnp.zeros((coords.shape[0], 0), dtype=coords.dtype)
        gradients_all = jnp.zeros((coords.shape[0], 0), dtype=coords.dtype)

        # Process full batches
        for batch_idx in range(num_full_batches):
            start_idx = batch_idx * self.num_pairs_batch_test
            subcoords = jax.lax.dynamic_slice(
                coords,
                (0, start_idx, 0, 0),  # start indices for all 4 dimensions
                (
                    coords.shape[0],
                    self.num_pairs_batch_test,
                    coords.shape[2],
                    coords.shape[3],
                ),  # slice sizes for all 4 dimensions
            )

            # Process batch
            outputs = self.apply_solver_times_grad_vel_jitted(
                solver_param, inputs=subcoords, p=p, a=a
            )
            pred_times, input_gradients, pred_vel = outputs

            pred_times = jnp.asarray(
                jax.lax.stop_gradient(pred_times), dtype=subcoords.dtype
            )
            pred_vel = jnp.asarray(
                jax.lax.stop_gradient(pred_vel), dtype=subcoords.dtype
            )
            input_gradients = jnp.asarray(
                jax.lax.stop_gradient(input_gradients), dtype=subcoords.dtype
            )

            # Concatenate results
            if batch_idx == 0:
                pred_times_all = pred_times
                pred_vel_all = pred_vel
                gradients_all = input_gradients
            else:
                pred_times_all = jnp.concatenate((pred_times_all, pred_times), axis=1)
                pred_vel_all = jnp.concatenate((pred_vel_all, pred_vel), axis=1)
                gradients_all = jnp.concatenate(
                    (gradients_all, input_gradients), axis=1
                )

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

            outputs = self.apply_solver_times_grad_vel_jitted(
                solver_param, inputs=remainder, p=p, a=a
            )
            pred_times, input_gradients, pred_vel = outputs

            pred_times = jnp.asarray(
                jax.lax.stop_gradient(pred_times), dtype=remainder.dtype
            )
            pred_vel = jnp.asarray(
                jax.lax.stop_gradient(pred_vel), dtype=remainder.dtype
            )
            input_gradients = jnp.asarray(
                jax.lax.stop_gradient(input_gradients), dtype=remainder.dtype
            )

            if num_full_batches > 0:
                pred_times_all = jnp.concatenate((pred_times_all, pred_times), axis=1)
                pred_vel_all = jnp.concatenate((pred_vel_all, pred_vel), axis=1)
                gradients_all = jnp.concatenate(
                    (gradients_all, input_gradients), axis=1
                )
            else:
                pred_times_all = pred_times
                pred_vel_all = pred_vel
                gradients_all = input_gradients

        return pred_times_all, pred_vel_all, gradients_all

    def _all_coords_no_plot(self, solver_param, coords, p, a):
        """Process all coordinates with plotting requirements (including gradients and velocity)"""
        # Only process times (no plotting)

        num_pairs_per_vel = coords.shape[1]

        # Calculate number of full batches and remainder information
        num_full_batches = num_pairs_per_vel // self.num_pairs_batch_test
        remainder_start = num_full_batches * self.num_pairs_batch_test
        has_remainder = remainder_start < num_pairs_per_vel

        # Initialize accumulators for gradients and velocity
        pred_times_all = jnp.zeros((coords.shape[0], 0), dtype=coords.dtype)

        # Process full batches
        for batch_idx in range(num_full_batches):
            start_idx = batch_idx * self.num_pairs_batch_test
            subcoords = jax.lax.dynamic_slice(
                coords,
                (0, start_idx, 0, 0),  # start indices for all 4 dimensions
                (
                    coords.shape[0],
                    self.num_pairs_batch_test,
                    coords.shape[2],
                    coords.shape[3],
                ),  # slice sizes for all 4 dimensions
            )

            # Process batch
            pred_times = jax.lax.stop_gradient(
                self.apply_solver_times_jitted(
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
                self.apply_solver_times_jitted(
                    solver_param,
                    inputs=remainder,
                    p=p,
                    a=a,
                )
            )

            pred_remainder = jnp.asarray(pred_remainder, dtype=remainder.dtype)

            if num_full_batches > 0:
                pred_times_all = jnp.concatenate(
                    (pred_times_all, pred_remainder), axis=1
                )
            else:
                pred_times_all = pred_remainder

        return pred_times_all

    def _validate_one_vel_against_gt(
        self,
        vel,
        vel_idx,
        p,
        a,
        solver_params,
        gt_dataset,
        visualize=False,
        name="val",
    ):
        """Validate model predictions against ground truth data"""
        fmm_comp_times = 0.0
        neural_inf_comp_times = 0.0

        # Get reference data and timing
        ref_times, fmmTime = gt_dataset[vel_idx]
        fmm_comp_times += fmmTime
        coords = gt_dataset.grid_data["coords"]

        # Handle visualization case
        if visualize:
            # Process coordinates for plotting (includes velocity and gradients)
            pred_times_all, pred_vel_all, gradients_all = self._all_coords_for_plot(
                solver_params, coords, p, a
            )

        # Time the actual inference
        start = time.time()
        pred_times_all = self._all_coords_no_plot(solver_params, coords, p, a)

        # Wait for computation to complete before timing
        pred_times_all.block_until_ready()
        neural_inf_comp_times = time.time() - start

        # Reshape predictions to match reference shape
        pred_times_all = pred_times_all.reshape(*ref_times.shape)

        # Calculate error metrics
        rmae = jnp.abs(pred_times_all - ref_times).sum() / jnp.abs(ref_times).sum()
        re = jnp.sqrt(((pred_times_all - ref_times) ** 2).sum() / (ref_times**2).sum())

        # Visualize if needed
        if visualize:
            ds_name = gt_dataset.name
            self.visualize_gt(
                gt_dataset.grid_data,
                vel,
                pred_vel_all.reshape(*ref_times.shape, 2),
                ref_times,
                pred_times_all,
                gradients_all.reshape(*ref_times.shape, -1),
                f"{name}/gt_comp_{ds_name}_{vel_idx}",
                all_indices=ds_name != "full",
            )

            if self.visualize_equiv is not None and ds_name != "top":
                self.visualize_equiv(
                    vel=vel,
                    pred_vel=pred_vel_all.reshape(*ref_times.shape, 2),
                    ref_times=ref_times,
                    pred_times=pred_times_all,
                    name=f"{name}/gt_comp_{ds_name}_{vel_idx}",
                    solver_fn=self.apply_solver_times_grad_vel_jitted,
                    solver_params=solver_params,
                    p=p,
                    a=a,
                )

        return rmae, re, fmm_comp_times, neural_inf_comp_times

    def update_prog_bar(self, step, train=True):
        """Update the progress bar.

        Args:
            desc: The description string.
            loss: The current loss.
            epoch: The current epoch.
            step: The current step.
        """
        # If we are at the beginning of the epoch, reset the progress bar
        if step == 0:
            # Depending on whether we are training or validating, set the total number of steps
            if train:
                self.prog_bar.total = len(self.train_loader)
            else:
                self.prog_bar.total = len(self.val_loader)
            self.prog_bar.reset()
        else:
            self.prog_bar.update(self.config.logging.log_every_n_steps)

        if train:
            epoch = self.epoch
            total_epochs = self.total_train_epochs
        else:
            epoch = self.cur_eval_epoch
            total_epochs = self.total_eval_epochs

        prog_bar_str = self.prog_bar_desc.format(
            state="Training" if train else "Eval",
            epoch=epoch,
            total_epochs=total_epochs,
        )

        # append metrics to description string
        if self.metrics:
            for k, v in self.metrics.items():
                prog_bar_str += f" -- {k} {v:.4f}"

        self.prog_bar.set_description_str(prog_bar_str)
