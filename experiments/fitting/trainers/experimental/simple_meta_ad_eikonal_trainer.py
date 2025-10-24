import jax.numpy as jnp
from functools import partial
import numpy as np
import optax
import jax
from experiments.fitting.trainers._base.base_eikonal_trainer import BaseEikonalTrainer
from experiments.fitting.utils.schedulers import cosine_diminishing_schedule
import time

from experiments.fitting.utils.logging import logger  # Add this line


class MetaAutoDecodingEikonalTrainer(BaseEikonalTrainer):
    """Meta-learning using meta-sgd trainer for AutoDecoding SNeF with optimizations for speed."""

    class TrainState(BaseEikonalTrainer.TrainState):
        reptile_opt_states: dict = None
        step_count: int = 0

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

        # Pre-compute learning rates
        self.meta_sgd_lrs = {
            "pose_pos": jnp.full((1,), self.config.meta.inner_learning_rate_poses),
            "appearance": jnp.full((1,), self.config.meta.inner_learning_rate_codes),
            "solver": jnp.full((1,), self.config.meta.inner_learning_rate_solver),
        }

        # Add orientation learning rate if we have orientation dimensions
        if self.config.geometry.dim_orientation > 0:
            self.meta_sgd_lrs["pose_ori"] = jnp.full(
                (1,), self.config.meta.inner_learning_rate_poses
            )

        # Calculate total training steps
        self.total_training_steps = self.total_train_epochs * len(self.train_loader)

    def _setup_optimizers(self):
        """Pre-compute optimizers, schedulers, and reptile components."""
        # Create LR schedule once for inner loop
        self.lr_schedule = self._create_lr_schedule()

        # Determine optimizer type for inner loop
        self.optimizer_type = self.config.meta.inner_optimizer.lower()

        # Pre-compute static optimizers for different parameter types (inner loop)
        self.optimizers = {}
        for param_name, lr in self.meta_sgd_lrs.items():
            self.optimizers[param_name] = self._create_optimizer_for_param(
                lr, param_name
            )

        # Create solver optimizer for outer loop
        if self.use_outer_scheduler:
            # Get parameters for cosine schedule
            base_lr = self.config.optimizer.learning_rate_solver
            min_lr_factor = self.config.optimizer.lr_scheduler.min_lr_factor
            warmup_steps = self.config.optimizer.lr_scheduler.warmup_steps
            freq = self.config.optimizer.lr_scheduler.lr_scheduler_freq

            # Create schedule function
            solver_schedule = cosine_diminishing_schedule(
                init_value=base_lr,
                min_value=base_lr * min_lr_factor,
                total_steps=self.total_training_steps,
                warmup_steps=warmup_steps,
                freq=freq,
            )

            # Create optimizer with schedule
            self.solver_opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                optax.scale_by_schedule(solver_schedule),
                optax.scale(-1.0),
            )
        else:
            # Standard optimizer without scheduling
            self.solver_opt = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                optax.scale(-self.config.optimizer.learning_rate_solver),
            )

        # Create Reptile optimizers if reptile is enabled
        if self.config.meta.reptile.enable:
            appearance_lr = self.config.optimizer.learning_rate_codes
            position_lr = self.config.optimizer.learning_rate_poses

            if self.use_outer_scheduler:
                # Get parameters for schedules
                min_lr_factor = self.config.optimizer.lr_scheduler.min_lr_factor
                warmup_steps = self.config.optimizer.lr_scheduler.warmup_steps
                freq = self.config.optimizer.lr_scheduler.lr_scheduler_freq

                # Create schedules
                appearance_schedule = cosine_diminishing_schedule(
                    init_value=appearance_lr,
                    min_value=appearance_lr * min_lr_factor,
                    total_steps=self.total_training_steps,
                    warmup_steps=warmup_steps,
                    freq=freq,
                )

                position_schedule = cosine_diminishing_schedule(
                    init_value=position_lr,
                    min_value=position_lr * min_lr_factor,
                    total_steps=self.total_training_steps,
                    warmup_steps=warmup_steps,
                    freq=freq,
                )

                # Create optimizers with schedules
                self.reptile_opts = {
                    "appearance": optax.chain(
                        optax.scale_by_schedule(appearance_schedule),
                        optax.sgd(learning_rate=1.0),
                    ),
                    "pose_pos": optax.chain(
                        optax.scale_by_schedule(position_schedule),
                        optax.sgd(learning_rate=1.0),
                    ),
                    "pose_ori": optax.chain(
                        optax.scale_by_schedule(position_schedule),
                        optax.sgd(learning_rate=1.0),
                    ),
                }
            else:
                # Standard optimizers without scheduling
                self.reptile_opts = {
                    "appearance": optax.sgd(learning_rate=appearance_lr),
                    "pose_pos": optax.sgd(learning_rate=position_lr),
                    "pose_ori": optax.sgd(learning_rate=position_lr),
                }

    def _create_lr_schedule(self):
        """Create learning rate schedule based on config."""
        sched_type = self.config.meta.lr_schedule_type
        initial_factor = self.config.meta.initial_lr_factor
        final_factor = self.config.meta.final_lr_factor

        # Number of training steps for inner loop
        steps = self.static_num_inner_steps

        if sched_type == "constant":
            return optax.constant_schedule(initial_factor)
        elif sched_type == "linear":
            return optax.linear_schedule(
                init_value=initial_factor,
                end_value=final_factor,
                transition_steps=steps,
            )
        elif sched_type == "cosine":
            return optax.cosine_decay_schedule(
                init_value=initial_factor, decay_steps=steps, alpha=final_factor
            )
        elif sched_type == "exponential":
            decay_rate = self.config.meta.lr_decay_rate
            return optax.exponential_decay(
                init_value=initial_factor,
                transition_steps=steps // 10 or 1,
                decay_rate=decay_rate,
                staircase=False,
            )
        elif sched_type == "cosine_diminishing":
            # Use our custom diminishing cosine schedule for the inner loop
            freq = getattr(self.config.meta, "lr_scheduler_freq", 1.0)
            warmup_steps = getattr(self.config.meta, "warmup_steps", 0)

            return cosine_diminishing_schedule(
                init_value=initial_factor,
                min_value=final_factor,
                total_steps=steps,
                warmup_steps=warmup_steps,
                freq=freq,
            )
        else:
            return optax.constant_schedule(initial_factor)

    def _create_optimizer_for_param(self, base_lr, name):
        """Create optimizer with learning rate schedule for a parameter type."""
        if self.optimizer_type == "adam":
            # Chain transformations correctly - apply the schedule to scale the gradient
            return optax.chain(
                optax.scale_by_adam(b1=0.0, b2=0.999, eps=1e-8),
                # Use the schedule to scale the gradient after Adam scaling
                optax.scale_by_schedule(self.lr_schedule),
                # Apply the base learning rate
                optax.scale(-base_lr),
            )
        else:
            # For SGD, chain the scale_by_schedule with the base learning rate scaling
            return optax.chain(
                optax.scale_by_schedule(self.lr_schedule), optax.scale(-base_lr)
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
            static_argnums=(3, 4),
        )

        # Pre-compile parameter updates for better performance
        self._update_params = jax.jit(
            lambda params, updates: optax.apply_updates(params, updates)
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

        # Initialize parameters
        if self.config.meta.reptile.enable:
            params = {"solver": solver_params, "init_latents": init_latents_params}
        else:
            params = {"solver": solver_params}

        # Initialize optimizer states
        solver_opt_state = self.solver_opt.init(solver_params)

        # Initialize reptile optimizer states if needed
        reptile_opt_states = None
        if self.config.meta.reptile.enable:
            reptile_opt_states = {
                "appearance": self.reptile_opts["appearance"].init(
                    init_latents_params["params"]["appearance"]
                ),
                "pose_pos": self.reptile_opts["pose_pos"].init(
                    init_latents_params["params"]["pose_pos"]
                ),
            }

            # Add orientation if it exists
            if "pose_ori" in init_latents_params["params"]:
                reptile_opt_states["pose_ori"] = self.reptile_opts["pose_ori"].init(
                    init_latents_params["params"]["pose_ori"]
                )

        # Create train state with step counter
        train_state = self.TrainState(
            params=params,
            solver_opt_state=solver_opt_state,
            reptile_opt_states=reptile_opt_states,
            step_count=0,
            rng=key,
        )

        return train_state

    def _loss_fn(
        self, params, sub_batch_coords, sub_true_vel, reduce_batch=True, train=True
    ):
        """Compute Eikonal loss and MSE for given parameters and coordinates."""
        # Get latent variables from autodecoder
        if train:
            autodecoder = self.train_autodecoder
        else:
            autodecoder = self.val_autodecoder

        p, a = autodecoder.apply(params["autodecoder"])

        # Compute predictions and loss
        outputs = self.solver.apply(
            params["solver"],
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
    @partial(jax.jit, static_argnums=(0, 8))
    def _process_batch(
        self,
        params,
        opt_states,
        batch_coords,
        true_vel,
        sub_batch_indices,
        step_idx,
        inner_autodecoder_params,
        train=True,
    ):
        """Process a single batch of data with Optax optimization.
        JIT-compiled for efficiency."""
        # Get batch data efficiently with a single take operation
        sub_coords = jnp.take(batch_coords, sub_batch_indices, axis=1)
        sub_vel = jnp.take(true_vel, sub_batch_indices, axis=1)

        # Compute gradients
        state_params = {
            "solver": params["solver"],
            "autodecoder": inner_autodecoder_params,
        }

        gradients, _ = self.inner_grad_fn(
            state_params,
            sub_batch_coords=sub_coords,
            sub_true_vel=sub_vel,
            reduce_batch=True,
            train=train,
        )

        # Extract gradients for autodecoder and solver
        # autodecoder_grads = jax.lax.stop_gradient(gradients["autodecoder"]["params"])

        autodecoder_grads = gradients["autodecoder"]["params"]

        # Update autodecoder parameters - make a copy of the original params structure
        updated_autodecoder_params = jax.tree_map(lambda x: x, inner_autodecoder_params)
        new_opt_states = dict(opt_states)  # Create copy to update

        # Update each parameter type separately using its own optimizer
        for param_name, optimizer in self.optimizers.items():
            if param_name != "solver":
                if (
                    param_name in autodecoder_grads
                    and param_name in inner_autodecoder_params["params"]
                ):
                    # Apply optimizer update
                    updates, new_opt_state = optimizer.update(
                        autodecoder_grads[param_name],
                        opt_states[param_name],
                        params=inner_autodecoder_params["params"][param_name],
                    )

                    # Apply updates to parameters
                    updated_param = self._update_params(
                        inner_autodecoder_params["params"][param_name], updates
                    )

                    # Store updated parameters and optimizer state
                    updated_autodecoder_params["params"][param_name] = updated_param
                    new_opt_states[param_name] = new_opt_state

        # Update solver parameters if in training mode
        if train and "solver" in self.optimizers and "solver" in gradients:
            # solver_grad = jax.lax.stop_gradient(gradients["solver"])
            solver_grad = gradients["solver"]
            updates, new_opt_state = self.optimizers["solver"].update(
                solver_grad, opt_states["solver"], params=params["solver"]
            )
            updated_solver_params = self._update_params(params["solver"], updates)
            new_opt_states["solver"] = new_opt_state
        else:
            # Keep original solver params during evaluation
            updated_solver_params = params["solver"]

        # Return updated parameters and optimizer states
        return updated_autodecoder_params, updated_solver_params, new_opt_states

    def inner_loop(
        self,
        outer_params,
        outer_state,
        batch,
        autodecoder,
        reptile_upd=False,
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

        # Initialize autodecoder parameters
        if self.config.meta.reptile.enable:
            # Use repeated init_latents for the inner loop when using Reptile
            inner_autodecoder_params = jax.tree_map(
                lambda p: jnp.repeat(p, batch_size, axis=0),
                outer_state.params["init_latents"],
            )
        else:
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

        # Create initial inner state for tracking params
        initial_state = outer_state.replace(
            params={
                "solver": outer_params["solver"],
                "autodecoder": inner_autodecoder_params,
            },
            rng=key,
        )

        # Initialize optimizer states
        opt_states = {}
        for param_name, optimizer in self.optimizers.items():
            if param_name == "solver" and train:
                opt_states[param_name] = optimizer.init(outer_params["solver"])
            elif (
                param_name != "solver"
                and param_name in inner_autodecoder_params["params"]
            ):
                opt_states[param_name] = optimizer.init(
                    inner_autodecoder_params["params"][param_name]
                )

        # Initialize current parameters
        curr_autodecoder_params = inner_autodecoder_params
        curr_solver_params = outer_params["solver"]
        curr_opt_states = opt_states
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
                curr_autodecoder_params, curr_solver_params, curr_opt_states = (
                    self._process_batch(
                        {"solver": curr_solver_params},
                        curr_opt_states,
                        batch_coords,
                        true_vel,
                        indices_for_batch,
                        step_idx,
                        curr_autodecoder_params,
                        train,
                    )
                )

            # Apply clipping if configured
            if self.config.geometry.clip_pos:
                curr_autodecoder_params = autodecoder.clip_pos(curr_autodecoder_params)

        # Final state
        curr_state = initial_state.replace(
            params={
                "solver": curr_solver_params,
                "autodecoder": curr_autodecoder_params,
            }
        )

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

        # IMPORTANT: First-order approximation - stop gradient on adapted parameters
        # first_order_solver_params = jax.lax.stop_gradient(curr_solver_params)

        first_order_solver_params = curr_solver_params

        # Replace with first-order parameters for meta-gradient computation
        first_order_state = curr_state.replace(
            params={
                "solver": first_order_solver_params,
                "autodecoder": curr_state.params["autodecoder"],
            }
        )

        # Compute final loss and MSE with first-order parameters
        final_loss, final_mse = self._loss_fn(
            first_order_state.params,
            sub_batch_coords=eval_coords,
            sub_true_vel=eval_vel,
            reduce_batch=False,
            train=train,
        )

        # Calculate improvements for Reptile if needed
        if reptile_upd:
            initial_loss, _ = self._loss_fn(
                initial_state.params,
                sub_batch_coords=eval_coords,
                sub_true_vel=eval_vel,
                reduce_batch=False,
                train=train,
            )
            improvements = jax.lax.stop_gradient(
                jnp.maximum(0.0, initial_loss - final_loss)
            )
            # improvements = jnp.ones_like(final_loss)
        else:
            improvements = None

        # Return with updated state
        return jnp.mean(final_loss), (
            final_mse,
            jax.lax.stop_gradient(curr_state.replace(rng=key)),
            improvements,
        )

    def train_update_solver(self, args, batch):
        """Perform solver update and Reptile update in the outer loop.

        Args:
            args: Tuple containing (state, grads, inner_state, improvements)
            batch: The current batch of data

        Returns:
            Tuple of (updated_params, solver_opt_state, updated_inner_state)
        """
        state_arg, grads_arg, inner_state_arg, improvements_arg = args

        # Update step counter for scheduling
        step_count = state_arg.step_count + 1

        # Update solver with Adam optimizer (with optional scheduling)
        solver_updates, solver_opt_state = self.solver_opt.update(
            grads_arg["solver"], state_arg.solver_opt_state, state_arg.step_count
        )
        solver_params = optax.apply_updates(state_arg.params["solver"], solver_updates)

        # Update inner state to reflect the solver update
        updated_inner_state = inner_state_arg.replace(
            params={
                "solver": solver_params,
                "autodecoder": inner_state_arg.params["autodecoder"],
            },
            solver_opt_state=solver_opt_state,
        )

        # If Reptile updates are enabled, apply them
        if self.config.meta.reptile.enable:
            # Calculate importance weights for each example based on improvements
            # Scale improvements by temperature for better weighting
            scaled_improvements = (
                improvements_arg / self.config.meta.reptile.weight_temperature
            )

            # Apply softmax to get normalized weights (with numerical stability)
            max_imp = jnp.max(scaled_improvements)
            exp_improvements = jnp.exp(scaled_improvements - max_imp)
            weights = exp_improvements / (jnp.sum(exp_improvements) + 1e-8)

            # Get adapted and initial parameters
            adapted_params = updated_inner_state.params["autodecoder"]
            init_param = state_arg.params["init_latents"]
            batch_size = batch[0].shape[0]

            # Expand initial parameters to match the batch size
            inner_autodecoder_params = jax.tree_map(
                lambda p: jnp.repeat(p, batch_size, axis=0),
                init_param,
            )

            # Calculate the weighted parameter differences (movement vectors)
            # First compute differences for each example
            diffs = jax.tree_map(
                lambda adapted, orig: adapted - orig,
                adapted_params,
                inner_autodecoder_params,
            )

            # Then weight by the improvement-based importances
            weighted_diffs = jax.tree_map(
                lambda p: jnp.sum(p * weights[:, None, None], axis=0, keepdims=True),
                diffs,
            )

            # Create a copy of the initial parameters to update
            reptile_params = jax.tree_map(lambda x: x, init_param)
            reptile_opt_states = state_arg.reptile_opt_states

            # Update appearance parameters with scheduled optimizer if needed
            appearance_updates, appearance_opt_state = self.reptile_opts[
                "appearance"
            ].update(
                weighted_diffs["params"]["appearance"],
                reptile_opt_states["appearance"],
                step_count,
            )
            reptile_params["params"]["appearance"] = optax.apply_updates(
                reptile_params["params"]["appearance"], appearance_updates
            )
            reptile_opt_states["appearance"] = appearance_opt_state

            # Update position parameters with scheduled optimizer if needed
            position_updates, position_opt_state = self.reptile_opts["pose_pos"].update(
                weighted_diffs["params"]["pose_pos"],
                reptile_opt_states["pose_pos"],
                step_count,
            )
            reptile_params["params"]["pose_pos"] = optax.apply_updates(
                reptile_params["params"]["pose_pos"], position_updates
            )
            reptile_opt_states["pose_pos"] = position_opt_state

            # Handle orientation if it exists
            if "pose_ori" in init_param["params"]:
                orientation_updates, orientation_opt_state = self.reptile_opts[
                    "pose_ori"
                ].update(
                    weighted_diffs["params"]["pose_ori"],
                    reptile_opt_states["pose_ori"],
                    step_count,
                )
                reptile_params["params"]["pose_ori"] = optax.apply_updates(
                    reptile_params["params"]["pose_ori"], orientation_updates
                )
                reptile_opt_states["pose_ori"] = orientation_opt_state

            if self.config.geometry.clip_pos:
                reptile_params = self.init_latents.clip_pos(reptile_params)

            # Return updated parameters including both solver and reptile updates
            params = {"solver": solver_params, "init_latents": reptile_params}

            # Return updated state with incremented step counter
            return (
                params,
                solver_opt_state,
                reptile_opt_states,
                step_count,
                updated_inner_state,
            )
        else:
            # If Reptile is disabled, just update the solver
            params = {"solver": solver_params}

            # Return updated state with incremented step counter
            return params, solver_opt_state, None, step_count, updated_inner_state

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
        outer_state = state.replace(rng=new_outer_key)

        # Conditional update based on whether we're training or evaluating
        if train:

            # Get gradients for the outer loop and update params
            (loss, (aux_data)), grads = jax.value_and_grad(
                self.inner_loop, has_aux=True
            )(
                {"solver": state.params["solver"]},
                outer_state=outer_state,
                autodecoder=autodecoder,
                batch=batch,
                reptile_upd=self.config.meta.reptile.enable,
                train=True,
            )

            # Unpack auxiliary data
            mse, inner_state, improvements = aux_data

            # Training update - update solver and apply Reptile if enabled
            if self.config.meta.reptile.enable:
                (
                    params,
                    solver_opt_state,
                    reptile_opt_states,
                    step_count,
                    inner_state,
                ) = self.train_update_solver(
                    (state, grads, inner_state, improvements), batch
                )

                # Create new state with updated values
                new_state = state.replace(
                    params=params,
                    solver_opt_state=solver_opt_state,
                    reptile_opt_states=reptile_opt_states,
                    step_count=step_count,
                    rng=new_outer_key,
                )
            else:
                params, solver_opt_state, _, step_count, inner_state = (
                    self.train_update_solver(
                        (state, grads, inner_state, improvements), batch
                    )
                )

                # Create new state with updated values
                new_state = state.replace(
                    params=params,
                    solver_opt_state=solver_opt_state,
                    step_count=step_count,
                    rng=new_outer_key,
                )
        else:

            loss, (aux_data) = self.inner_loop(
                {"solver": state.params["solver"]},
                outer_state=outer_state,
                autodecoder=autodecoder,
                batch=batch,
                reptile_upd=False,
                train=False,
            )
            loss = jax.lax.stop_gradient(loss)

            # Unpack auxiliary data
            mse, inner_state, improvements = aux_data

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

        # Metrics for timing and error tracking
        if gt_dataset:
            total_neural_fit_comp_times = 0.0
            total_rmae = {ds.name: 0.0 for ds in gt_dataset}
            total_re = {ds.name: 0.0 for ds in gt_dataset}
            total_fmm_comp_times = {ds.name: 0.0 for ds in gt_dataset}
            total_neural_inf_comp_times = {ds.name: 0.0 for ds in gt_dataset}
        else:

            total_neural_fit_comp_times = None
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

            if visualize:
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
                            visualize=visualize,
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
        # self.metrics[f"{name}_neural_fit_comp_times"] = total_neural_fit_comp_times

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
                self.cur_val_metric = self.metrics["val_eiko_epoch"]

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
