import jax.numpy as jnp
from functools import partial
import numpy as np
import optax
import jax
from experiments.fitting.trainers._base.base_eikonal_trainer import BaseEikonalTrainer
from experiments.fitting.utils.schedulers import cosine_diminishing_schedule
import time

from experiments.fitting.utils.logging import logger  # Add this line
from flax import struct


class MetaAutoDecodingEikonalTrainer(BaseEikonalTrainer):
    """Meta-learning using meta-sgd trainer for AutoDecoding SNeF with optimizations for speed."""

    class TrainState(BaseEikonalTrainer.TrainState):
        meta_sgd_opt_state: optax.OptState = struct.field(pytree_node=True)
        step_count: int = 0
        vmin: float = 0.0
        vmax: float = 1.0

    def __init__(
        self,
        config,
        solver,
        init_latents,
        train_autodecoder,
        val_autodecoder,
        test_autodecoder,
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
            solver=solver,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            seed=seed,
            visualize_reconstruction=visualize_reconstruction,
            gt_dataset_val=gt_dataset_val,
            gt_dataset_test=gt_dataset_test,
            visualize_gt=visualize_gt,
            visualize_equiv=visualize_equiv,
            num_epochs=num_epochs,
        )

        self.cur_eval_epoch = 0
        self.total_eval_epochs = 0

        # Flag for using cosine scheduler
        self.use_outer_scheduler = self.config.optimizer.lr_scheduler.activate

        # Set autodecoders
        self.init_latents = init_latents
        self.train_autodecoder = train_autodecoder
        self.val_autodecoder = val_autodecoder
        self.test_autodecoder = test_autodecoder

        # Pre-compute static values for efficiency
        self._setup_static_values()

        # Pre-compute optimizers and schedulers
        self._setup_optimizers()

    def _setup_static_values(self):
        """Pre-compute static values used throughout training to avoid recomputation."""
        self.static_num_pairs_total = int(self.config.data.n_coords / 2)
        self.static_num_pairs_meta = int(self.config.meta.num_pairs)
        self.static_num_pairs_mini_batch = int(self.config.data.num_pairs)

        self.static_num_mini_batches = (
            self.static_num_pairs_meta // self.static_num_pairs_mini_batch
        )
        self.static_num_inner_steps = int(self.config.meta.num_inner_steps)
        self.static_indices = jnp.arange(self.static_num_pairs_total)

        # Calculate total training steps
        self.total_training_steps = self.total_train_epochs * len(self.train_loader)

    def _setup_optimizers(self):
        """Pre-compute optimizers, schedulers, and reptile components."""

        # Create solver optimizer for outer loop
        if self.use_outer_scheduler:
            # Get parameters for cosine schedule
            min_lr_factor = self.config.optimizer.lr_scheduler.min_lr_factor
            warmup_steps = self.config.optimizer.lr_scheduler.warmup_steps
            freq = self.config.optimizer.lr_scheduler.lr_scheduler_freq

            # Create schedule function
            solver_schedule = cosine_diminishing_schedule(
                init_value=self.config.optimizer.learning_rate_solver,
                min_value=self.config.optimizer.learning_rate_solver * min_lr_factor,
                total_steps=self.total_training_steps,
                warmup_steps=warmup_steps,
                freq=freq,
            )

            # Create optimizer with schedule
            self.solver_opt = optax.chain(
                optax.clip_by_global_norm(self.config.training.gradient_clip_val),
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                optax.scale_by_schedule(solver_schedule),
                optax.scale(-1.0),
            )

            meta_sgd_schedule = cosine_diminishing_schedule(
                init_value=self.config.meta.learning_rate_meta_sgd,
                min_value=self.config.meta.learning_rate_meta_sgd * min_lr_factor,
                total_steps=self.total_training_steps,
                warmup_steps=warmup_steps,
                freq=freq,
            )

            self.meta_sgd_opt = optax.chain(
                optax.clip_by_global_norm(self.config.training.gradient_clip_val),
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                optax.scale_by_schedule(meta_sgd_schedule),
                optax.scale(-1.0),
            )
        else:
            # Standard optimizer without scheduling
            self.solver_opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                optax.scale(-self.config.optimizer.learning_rate_solver),
            )

            # Standard optimizer without scheduling
            self.meta_sgd_opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                optax.scale(-self.config.meta.learning_rate_meta_sgd),
            )

    def create_functions(self):
        """Creates and JIT-compiles the training and validation functions with static arguments."""
        super().create_functions()

        # Optimize train and val steps with static_argnums for faster compilation
        self.train_step = self._create_optimized_outer_step(
            self.train_autodecoder,
            train=True,
        )

        self.val_step = self._create_optimized_outer_step(
            self.val_autodecoder,
            train=False,
        )
        self.test_step = self._create_optimized_outer_step(
            self.test_autodecoder,
            train=False,
        )

        self.validate_epoch = lambda state: self.eval_epoch(state, name="val")
        self.test_epoch = lambda state: self.eval_epoch(state, name="test")

        # Optimize inner gradient computation with jit
        self.inner_grad_fn = jax.jit(
            jax.grad(
                self._loss_fn,
                has_aux=True,
            ),
            static_argnums=(4, 5),
        )

    def _create_optimized_outer_step(
        self,
        autodecoder,
        train,
    ):
        """Creates an optimized and jitted outer step function."""
        # Use partial to bind static arguments
        outer_step_fn = partial(
            self.outer_step,
            autodecoder=autodecoder,
            train=train,
        )

        # JIT compile with donation for memory efficiency
        return jax.jit(
            outer_step_fn,
            # donate_argnums=(0,),  # Donate the input state for memory efficiency
        )

    def init_train_state(self):
        """Initialize training state with pre-compiled optimizers."""
        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key efficiently
        key, solver_key, init_latents_key = jax.random.split(key, 3)

        # Create a test batch to get the shape of the latent space
        init_latents_params = self.init_latents.init(init_latents_key)
        p, a = self.init_latents.apply(init_latents_params)

        # Initialize solver with more efficient batched setup
        sample_coords = jax.random.normal(
            solver_key,
            (
                self.config.data.train_batch_size,
                self.config.data.num_pairs,
                2,
                self.config.geometry.dim_signal,
            ),
        )
        solver_params = self.solver.init(solver_key, sample_coords, p, a)

        # Initialize optimizer states
        solver_opt_state = self.solver_opt.init(solver_params)

        # Initialize learning rates for the autodecoder
        lr_pos = jnp.ones((1)) * self.config.meta.inner_learning_rate_poses
        lr_a = jnp.ones((a.shape[-1])) * self.config.meta.inner_learning_rate_codes

        # Put lrs in frozendict
        meta_sgd_lrs = {
            "pose_pos": lr_pos,
            "appearance": lr_a,
        }

        # Add orientation learning rate if we have orientation dimensions
        if p[1] is not None:
            lr_ori = jnp.ones((1)) * self.config.meta.inner_learning_rate_poses
            meta_sgd_lrs["pose_ori"] = lr_ori

        # Create train state with step counter
        train_state = self.TrainState(
            params={
                "solver": solver_params,
                "meta_sgd_lrs": meta_sgd_lrs,
            },
            solver_opt_state=solver_opt_state,
            meta_sgd_opt_state=self.meta_sgd_opt.init(meta_sgd_lrs),
            step_count=0,
            rng=key,
            vmin=self.solver.vmin,
            vmax=self.solver.vmax,
        )

        return train_state

    def _loss_fn(
        self,
        params_auto,
        params_solver,
        sub_batch_coords,
        sub_true_vel,
        reduce_batch=True,
        train=True,
    ):
        """Compute Eikonal loss and MSE for given parameters and coordinates."""
        # Get latent variables from autodecoder
        if train:
            autodecoder = self.train_autodecoder
        else:
            autodecoder = self.val_autodecoder

        p, a = autodecoder.apply(params_auto)

        # Compute predictions and loss
        outputs = self.solver.apply(
            params_solver,
            method=self.solver.times_grad_vel,
            inputs=sub_batch_coords,
            p=p,
            a=a,
            aux_vel=True,
        )

        # Unpack outputs
        pred_times, input_grads, norm_input_grad, pred_vel = outputs
        # Compute Eikonal loss
        loss = self.eiko_loss(norm_input_grad, sub_true_vel, reduce_batch=reduce_batch)

        # Compute MSE (stop gradient to avoid computing unnecessary gradients)
        mse = jax.lax.stop_gradient(jnp.sum((pred_vel - sub_true_vel) ** 2))

        return loss, mse

    # Pre-compile batch processor to avoid recompilation in inner loop
    @partial(jax.jit, static_argnums=(0, 6))
    def _process_batch(
        self,
        outer_params,
        inner_autodecoder_params,
        batch_coords,
        true_vel,
        sub_batch_indices,
        train=True,
    ):
        """Process a single batch of data with Optax optimization.
        JIT-compiled for efficiency."""
        # Get batch data efficiently with a single take operation
        sub_coords = jnp.take(batch_coords, sub_batch_indices, axis=1)
        sub_vel = jnp.take(true_vel, sub_batch_indices, axis=1)

        # Compute gradients
        autodecoder_grads, _ = self.inner_grad_fn(
            inner_autodecoder_params,
            params_solver=outer_params["solver"],
            sub_batch_coords=sub_coords,
            sub_true_vel=sub_vel,
            reduce_batch=True,
            train=train,
        )

        # Update autodecoder parameters - make a copy of the original params structure
        updated_autodecoder_params = jax.tree_map(lambda x: x, inner_autodecoder_params)

        # jax.debug.print("Value: {x}", x=grad_params)
        if "appearance" in autodecoder_grads["params"]:
            updated_autodecoder_params["params"]["appearance"] = (
                updated_autodecoder_params["params"]["appearance"]
                - outer_params["meta_sgd_lrs"]["appearance"]
                * autodecoder_grads["params"]["appearance"]
            )

        if "pose_pos" in autodecoder_grads["params"]:
            updated_autodecoder_params["params"]["pose_pos"] = (
                updated_autodecoder_params["params"]["pose_pos"]
                - outer_params["meta_sgd_lrs"]["pose_pos"]
                * autodecoder_grads["params"]["pose_pos"]
            )

        if (
            "pose_ori" in outer_params["meta_sgd_lrs"]
            and "pose_ori" in autodecoder_grads["params"]
        ):
            updated_autodecoder_params["params"]["pose_ori"] = (
                updated_autodecoder_params["params"]["pose_ori"]
                - outer_params["meta_sgd_lrs"]["pose_ori"]
                * autodecoder_grads["params"]["pose_ori"]
            )

        # Return updated parameters and optimizer states
        return updated_autodecoder_params

    def inner_loop(
        self,
        outer_params,
        outer_state,
        batch,
        autodecoder,
        train=True,
    ):
        """Memory-efficient implementation using pre-computed optimizers and schedulers.

        Args:
            outer_params: Parameters from the outer loop
            outer_state: State from the outer loop
            batch: Training batch
            autodecoder: Autodecoder model
            reptile_upd: Whether to do reptile updates
            train: Whether this is training (True) or evaluation (False)
        """
        # Unpack batch
        true_vel, batch_coords, vel_idx = batch
        batch_size = batch_coords.shape[0]

        # Initialize key for randomness
        key, init_key = jax.random.split(outer_state.rng)

        # Initialize fresh parameters when not using Reptile
        inner_autodecoder_params = autodecoder.init(init_key)

        # Handle noise addition if configured
        if self.config.meta.noise_pos_inner_loop:
            key, noise_key = jax.random.split(key)
            inner_autodecoder_params, _ = autodecoder.add_noise(
                inner_autodecoder_params,
                self.config.meta.noise_pos_inner_loop,
                noise_key,
            )

            # Apply position clipping if configured
            if self.config.geometry.clip_pos:
                inner_autodecoder_params = autodecoder.clip_pos(
                    inner_autodecoder_params
                )

        # Initialize current parameters
        curr_autodecoder_params = inner_autodecoder_params
        curr_key = key

        # Run the inner optimization loop
        for step_idx in range(self.static_num_inner_steps):
            # Generate batch indices
            perm_key, curr_key = jax.random.split(curr_key)
            batch_indices = jax.random.choice(
                perm_key,
                self.static_indices,
                shape=(self.static_num_pairs_meta,),
                replace=False,
            )

            # Process each mini-batch efficiently
            for batch_idx in range(self.static_num_mini_batches):
                # Get batch indices
                start_idx = batch_idx * self.static_num_pairs_mini_batch
                end_idx = start_idx + self.static_num_pairs_mini_batch
                indices_for_batch = batch_indices[start_idx:end_idx]

                # Process batch
                curr_autodecoder_params = self._process_batch(
                    outer_params,
                    curr_autodecoder_params,
                    batch_coords,
                    true_vel,
                    indices_for_batch,
                    train,
                )

            # Apply clipping if configured
            if self.config.geometry.clip_pos:
                curr_autodecoder_params = autodecoder.clip_pos(curr_autodecoder_params)

        # Evaluate final performance
        eval_key, key = jax.random.split(curr_key)
        eval_indices = jax.random.choice(
            eval_key,
            self.static_indices,
            shape=(self.static_num_pairs_mini_batch,),
            replace=False,
        )

        # Get evaluation data
        eval_coords = jnp.take(batch_coords, eval_indices, axis=1)
        eval_vel = jnp.take(true_vel, eval_indices, axis=1)

        # Compute final loss and MSE with first-order parameters
        final_loss, final_mse = self._loss_fn(
            params_auto=curr_autodecoder_params,
            params_solver=outer_params["solver"],
            sub_batch_coords=eval_coords,
            sub_true_vel=eval_vel,
            reduce_batch=False,
            train=train,
        )

        # Return with updated state
        return jnp.mean(final_loss), (
            final_mse,
            outer_state.replace(
                params={
                    "solver": outer_params["solver"],
                    "meta_sgd_lrs": outer_params["meta_sgd_lrs"],
                    "autodecoder": curr_autodecoder_params,
                },
                rng=key,
            ),
        )

    def train_update(self, state, grads, inner_state):
        """Perform solver update and Reptile update in the outer loop.

        Args:
            args: Tuple containing (state, grads, inner_state, improvements)
            batch: The current batch of data

        Returns:
            Tuple of (updated_params, solver_opt_state, updated_inner_state)
        """

        # Update step counter for scheduling

        # Update solver with Adam optimizer (with optional scheduling)
        solver_updates, solver_opt_state = self.solver_opt.update(
            grads["solver"], state.solver_opt_state, state.step_count
        )
        solver_params = optax.apply_updates(state.params["solver"], solver_updates)

        meta_sgd_lr_updates, meta_sgd_opt_state = self.meta_sgd_opt.update(
            grads["meta_sgd_lrs"], state.meta_sgd_opt_state, state.step_count
        )
        meta_sgd_lrs = optax.apply_updates(
            state.params["meta_sgd_lrs"], meta_sgd_lr_updates
        )

        # Clip meta_sgd_lrs between 1e-6 and 10
        meta_sgd_lrs = jax.tree_map(lambda x: jnp.clip(x, 1e-6, 50), meta_sgd_lrs)

        # Update inner state to reflect the solver update
        updated_inner_state = inner_state.replace(
            params={
                "solver": solver_params,
                "autodecoder": inner_state.params["autodecoder"],
            },
            solver_opt_state=solver_opt_state,
        )

        # Create new state with updated values
        new_state = state.replace(
            params={"solver": solver_params, "meta_sgd_lrs": meta_sgd_lrs},
            solver_opt_state=solver_opt_state,
            meta_sgd_opt_state=meta_sgd_opt_state,
            step_count=state.step_count + 1,
        )

        # Return updated state with incremented step counter
        return (
            new_state,
            updated_inner_state,
        )

    def outer_step(self, state, batch, autodecoder, train=True):
        """Performs a single outer-loop training step with optimized memory usage.

        Args:
            state: Current training state
            batch: Batch of training data
            autodecoder: Autodecoder model
            train: Whether this is training (True) or evaluation (False)

        Returns:
            Tuple of (loss, mse, (new_state, inner_state))
        """
        # Split random key for inner and outer loop randomness
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=inner_key)

        # Conditional update based on whether we're training or evaluating
        if train:

            # Get gradients for the outer loop and update params
            (loss, (mse, inner_state)), grads = jax.value_and_grad(
                self.inner_loop, has_aux=True
            )(
                state.params,
                outer_state=outer_state,
                autodecoder=autodecoder,
                batch=batch,
                train=True,
            )

            new_state, inner_state = self.train_update(state, grads, inner_state)
            new_state = new_state.replace(rng=new_outer_key)

        else:

            loss, (mse, inner_state) = self.inner_loop(
                state.params,
                outer_state=outer_state,
                autodecoder=autodecoder,
                batch=batch,
                train=False,
            )
            loss = jax.lax.stop_gradient(loss)

            # Evaluation - don't update parameters
            new_state = state

        # Return loss, MSE, and updated states
        return loss, mse, (new_state, inner_state)

    def eval_epoch(self, state, name="val"):
        """Optimized validation method with reduced overhead."""

        # Select appropriate components based on evaluation type
        if name == "val":
            autodecoder = self.val_autodecoder
            loader = self.val_loader
            gt_dataset = self.gt_dataset_val
            step_fn = self.val_step
        else:
            autodecoder = self.test_autodecoder
            loader = self.test_loader
            gt_dataset = self.gt_dataset_test
            step_fn = self.test_step

        # Initialize evaluation metrics
        eval_state = state
        losses = 0
        mse_tot = 0
        total_eval_points = 0
        self.total_eval_epochs = 1

        # NEW: Critical - we'll use one consistent step for all validation logs
        # This should be the current global step from training
        validation_log_step = self.global_step

        # Set up local counter for batches
        local_batch_idx = 0
        self.global_eval_step = 0
        total_neural_fit_comp_times = 0.0

        # Metrics for timing and error tracking
        if gt_dataset:
            total_rmae = {ds.name: 0.0 for ds in gt_dataset}
            total_re = {ds.name: 0.0 for ds in gt_dataset}
            total_fmm_comp_times = {ds.name: 0.0 for ds in gt_dataset}
            total_neural_inf_comp_times = {ds.name: 0.0 for ds in gt_dataset}
        else:
            total_rmae = None
            total_re = None
            total_fmm_comp_times = None
            total_neural_tot_comp_times = None
            total_neural_inf_comp_times = None

        # Process each batch in the evaluation loader
        for batch_idx, batch in enumerate(loader):
            # Measure fitting time
            start = time.time()
            loss, mse, (eval_state, inner_state) = step_fn(eval_state, batch)
            # Force computation of results
            loss_np = float(jax.device_get(loss))

            fit_time = time.time() - start

            mse_np = float(jax.device_get(mse))

            # Update metrics
            total_neural_fit_comp_times += fit_time
            num_points_batch = batch[0].shape[0] * batch[0].shape[1] * 2.0
            total_eval_points += num_points_batch
            losses += loss_np
            mse_tot += mse_np

            if batch_idx % self.config.logging.log_every_n_steps == 0:
                # Add metrics to buffer without committing yet
                logger.add_to_buffer(
                    {f"{name}_loss": loss_np, f"{name}_mse": mse_np / num_points_batch},
                    commit=False,
                )

                self.update_prog_bar(step=batch_idx, train=False)

            # Only visualize when necessary
            visualize = (
                not self.config.logging.debug
                and self.global_eval_step % self.config.logging.visualize_every_n_steps
                == 0
            )

            if visualize and self.visualize_reconstruction is not None:
                inner_state = self.visualize_batch(
                    inner_state,
                    batch,
                    name=f"{name}/recon-fitting",
                    base_dataset=loader.dataset.base_dataset,
                    autodecoder=autodecoder,
                )

            # GT validation when dataset is available
            if gt_dataset:
                for ds in gt_dataset:
                    name_ds = ds.name
                    true_vel, batch_coords, vel_idx = batch
                    # More efficient data gathering with list comprehension
                    true_full_vel = np.stack(
                        [loader.dataset.base_dataset[idx][0] for idx in vel_idx]
                    )
                    # Run GT validation
                    rmae, re, fmm_comp_times, neural_inf_comp_times = (
                        self.validate_against_gt(
                            autodecoder=autodecoder,
                            state=inner_state,
                            gt_dataset=ds,
                            batch_true_vel=true_full_vel,
                            batch_vel_idx=vel_idx,
                            visualize=visualize and self.visualize_gt is not None,
                            name=name,
                        )
                    )

                    # Update metrics
                    total_rmae[name_ds] += float(jax.device_get(rmae))
                    total_re[name_ds] += float(jax.device_get(re))
                    total_fmm_comp_times[name_ds] += fmm_comp_times
                    total_neural_inf_comp_times[name_ds] += neural_inf_comp_times

            self.global_eval_step += 1

        # Reset counters
        self.total_eval_epochs = 0
        self.global_eval_step = 0

        # Update final metrics
        self.metrics[f"{name}_eiko_epoch"] = losses / len(loader)
        self.metrics[f"{name}_mse_epoch"] = mse_tot / total_eval_points
        self.metrics[f"{name}_neural_fit_comp_times"] = total_neural_fit_comp_times

        # Batch log to wandb rather than individual logs
        log_data = {
            f"{name}_eiko_epoch": self.metrics[f"{name}_eiko_epoch"],
            f"{name}_mse_epoch": self.metrics[f"{name}_mse_epoch"],
            f"{name}_neural_fit_comp_times": total_neural_fit_comp_times,
        }

        # Add GT metrics if available
        if gt_dataset:
            for ds in gt_dataset:
                name_ds = ds.name
                self.metrics[f"{name}_{name_ds}_re"] = total_re[name_ds] / len(loader)
                log_data.update(
                    {
                        f"{name}_{name_ds}_rmae": total_rmae[name_ds] / len(loader),
                        f"{name}_{name_ds}_re": total_re[name_ds] / len(loader),
                        f"{name}_{name_ds}_neural_inf_comp_times": total_neural_inf_comp_times[
                            name_ds
                        ],
                        f"{name}_{name_ds}_neural_tot_comp_times": total_neural_fit_comp_times
                        + total_neural_inf_comp_times[name_ds],
                        f"{name}_{name_ds}_fmm_comp_times": total_fmm_comp_times[
                            name_ds
                        ],
                    }
                )

        if name == "val":
            if "val_top_re" in self.metrics:
                self.cur_val_metric = self.metrics["val_top_re"]
            elif "val_full_re" in self.metrics:
                self.cur_val_metric = self.metrics["val_full_re"]
            else:
                self.cur_val_metric = self.metrics["train_eiko_epoch"]

        # Log all metrics at once with explicit commit
        logger.log(log_data, step=validation_log_step, commit=True)

        return state, eval_state

    # Optimized GT validation with better parallelization
    def validate_against_gt(
        self,
        autodecoder,
        state,
        gt_dataset,
        batch_true_vel,
        batch_vel_idx,
        visualize=False,
        name="val",
    ):
        # Apply autodecoder once for the whole batch
        (batch_p_pos, batch_p_pose), batch_a = autodecoder.apply(
            state.params["autodecoder"]
        )

        # Define a function to process a single velocity
        def process_single_vel(args):
            i, vel_idx, vel = args
            p_pos = batch_p_pos[i][None, ...]

            if batch_p_pose is not None:
                p = (p_pos, batch_p_pose[i][None, ...])
            else:
                p = (p_pos, None)

            a = batch_a[i][None, ...]

            # Validate single velocity
            return self._validate_one_vel_against_gt(
                vel,
                vel_idx,
                p,
                a,
                gt_dataset=gt_dataset,
                solver_params=state.params["solver"],
                visualize=visualize,
                name=name,
            )

        total_rmae, total_re, total_fmm_times, total_neural_times = 0.0, 0.0, 0.0, 0.0

        for i, (vel_idx, vel) in enumerate(zip(batch_vel_idx, batch_true_vel)):
            # Process each velocity
            rmae, re, fmm_time, neural_time = process_single_vel((i, vel_idx, vel))

            # Update metrics
            total_rmae += rmae
            total_re += re
            total_fmm_times += fmm_time
            total_neural_times += neural_time

        return total_rmae, total_re, total_fmm_times, total_neural_times

    # Optimized visualization with reduced overhead
    def visualize_batch(self, state, batch, name, base_dataset, autodecoder=None):
        """More efficient batch visualization."""
        # Early return if visualization is unnecessary
        if not self.visualize_reconstruction:
            return state

        true_vel, batch_coords, vel_idx = batch

        if autodecoder is None:
            autodecoder = self.train_autodecoder

        # Apply autodecoder only once
        p, a = autodecoder.apply(state.params["autodecoder"])

        # More efficient data gathering
        true_full_vel = np.stack([base_dataset[idx][0] for idx in vel_idx])

        # Split random key only once
        rng, rng_plot = jax.random.split(state.rng)

        # Perform visualization
        self.visualize_reconstruction(
            targets=true_full_vel,
            solver_fn=self.apply_solver_velocities_jitted,
            solver_params=state.params["solver"],
            coords=batch_coords,
            rng=rng_plot,
            p=p,
            a=a,
            name=name,
        )

        return state.replace(rng=rng)
