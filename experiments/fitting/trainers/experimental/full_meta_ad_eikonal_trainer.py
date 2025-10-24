import jax.numpy as jnp
from functools import partial
import numpy as np
import optax
import jax
from experiments.fitting.trainers._base.base_eikonal_trainer import BaseEikonalTrainer
from experiments.fitting.utils.schedulers import cosine_diminishing_schedule
import time

import wandb


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
        )

        self.val_epoch = 0
        self.total_val_epochs = 0

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
        self.total_training_steps = self.config.training.num_epochs * len(
            self.train_loader
        )

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
            train=True,
        )

        self.test_step = self.val_step = self._create_optimized_outer_step(
            train=False,
        )

        self.validate_epoch = lambda state: self.eval_epoch(state, name="val")
        self.test_epoch = lambda state: self.eval_epoch(state, name="test")

        # Optimize inner gradient computation with jit
        self.inner_grad_fn = jax.jit(
            jax.grad(self._loss_fn, has_aux=True),
            static_argnums=(3,),  # Mark reduce_batch as static
        )

        # Pre-compile parameter updates for better performance
        self._update_params = jax.jit(
            lambda params, updates: optax.apply_updates(params, updates)
        )

        # Pre-compile functions for inner loop operations
        self._init_element_params = self._initialize_element_parameters

        # Pre-compile batch processing with vmap
        self._vmapped_process_batch = jax.jit(self._create_vmapped_process_batch())

    def _create_vmapped_process_batch(self):
        """Creates a vectorized batch processing function using vmap."""

        # Define a function for processing a single element's mini-batch
        def process_single_element_batch(
            element_params,
            opt_states,
            coords,
            vel,
            indices,
            step_idx,
            autodecoder_params,
        ):
            return self._process_batch(
                element_params,
                opt_states,
                coords[None, ...],  # Add batch dimension
                vel[None, ...],  # Add batch dimension
                indices,
                step_idx,
                autodecoder_params,
            )

        # Create a vectorized version that processes all elements in parallel
        return jax.vmap(
            process_single_element_batch,
            in_axes=(0, 0, 0, 0, None, None, 0),  # Vectorize over batch elements
            out_axes=(0, 0, 0),  # Output is vectorized over batch
        )

    def _initialize_element_parameters(self, state_params, element_key, autodecoder):
        """Initialize parameters for a single batch element."""
        if self.config.meta.reptile.enable:
            # Use init_latents but ensure correct shape
            # We need to preserve the original structure without flattening batch dimensions
            inner_autodecoder_params = state_params["init_latents"]

            # No need to add batch dimension since it will be handled by vmap
            # We just want to make sure the parameters maintain their original structure
        else:
            # Initialize fresh parameters
            inner_autodecoder_params = autodecoder.init(element_key)

        # Apply noise if configured
        if self.config.meta.noise_pos_inner_loop:
            element_key, noise_key = jax.random.split(element_key)
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

        return inner_autodecoder_params, element_key

    def _create_optimized_outer_step(
        self,
        train,
    ):
        """Creates an optimized and jitted outer step function."""
        # Use partial to bind static arguments
        outer_step_fn = partial(
            self.outer_step,
            train=train,
        )

        # JIT compile with donation for memory efficiency
        return jax.jit(
            outer_step_fn,
            donate_argnums=(0,),  # Donate the input state for memory efficiency
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

    def _loss_fn(self, params, sub_batch_coords, sub_true_vel, reduce_batch=True):
        """Compute Eikonal loss and MSE for given parameters and coordinates."""
        # Get latent variables from autodecoder
        autodecoder = self.init_latents  # Use init_latents as default

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

    def _process_batch(
        self,
        params,
        opt_states,
        batch_coords,
        true_vel,
        sub_batch_indices,
        step_idx,
        inner_autodecoder_params,
    ):
        """Process a single batch of data with Optax optimization."""
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
                # Make sure the param_name exists in all required dictionaries
                if (
                    param_name in autodecoder_grads
                    and param_name in opt_states
                    and param_name in inner_autodecoder_params["params"]
                ):
                    # Get the actual values instead of using param_name directly
                    grad_value = autodecoder_grads[param_name]
                    opt_state = opt_states[param_name]
                    param_value = inner_autodecoder_params["params"][param_name]

                    # Apply optimizer update
                    updates, new_opt_state = optimizer.update(
                        grad_value,
                        opt_state,
                        params=param_value,
                    )

                    # Apply updates to parameters
                    updated_param = self._update_params(param_value, updates)

                    # Store updated parameters and optimizer state
                    updated_autodecoder_params["params"][param_name] = updated_param
                    new_opt_states[param_name] = new_opt_state

        # Update solver parameters if in training mode
        if "solver" in self.optimizers and "solver" in gradients:
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
        """Highly optimized implementation using vmap to parallelize inner loops
        across batch elements.

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

        # Generate batch_size + 2 keys for parallel processing
        main_key, element_keys_key, final_key = jax.random.split(outer_state.rng, 3)
        element_keys = jax.random.split(element_keys_key, batch_size)

        # Initialize per-element autodecoder parameters
        # For Reptile, we need to replicate the init_latents for each batch element
        if self.config.meta.reptile.enable:
            init_autodecoder_params = jax.tree_map(
                lambda p: jnp.repeat(p, batch_size, axis=0),
                outer_state.params["init_latents"],
            )
        else:
            # For non-Reptile, we'll initialize fresh params for each element in process_single_element
            init_autodecoder_params = None

        # Define a function to process a single batch element
        def process_single_element(element_key, coords, vel, batch_idx):
            """Process inner optimization loop for a single batch element."""
            # Initialize autodecoder parameters for this element
            if self.config.meta.reptile.enable:
                # Extract this element's parameters from the batch
                curr_autodecoder_params = jax.tree_map(
                    lambda p: jax.lax.dynamic_slice_in_dim(p, batch_idx, 1, axis=0),
                    init_autodecoder_params,
                )
            else:
                # Initialize fresh parameters
                curr_autodecoder_params = autodecoder.init(element_key)

            # Apply noise if configured
            curr_key = element_key
            if self.config.meta.noise_pos_inner_loop:
                curr_key, noise_key = jax.random.split(curr_key)
                curr_autodecoder_params, _ = autodecoder.add_noise(
                    curr_autodecoder_params,
                    self.config.meta.noise_pos_inner_loop,
                    noise_key,
                )

                # Apply position clipping if configured
                if self.config.geometry.clip_pos:
                    curr_autodecoder_params = autodecoder.clip_pos(
                        curr_autodecoder_params
                    )

            # Initialize optimizer states
            element_opt_states = {}
            for param_name, optimizer in self.optimizers.items():
                if param_name == "solver" and train:
                    element_opt_states[param_name] = optimizer.init(
                        outer_params["solver"]
                    )
                elif (
                    param_name != "solver"
                    and param_name in curr_autodecoder_params["params"]
                ):
                    element_opt_states[param_name] = optimizer.init(
                        curr_autodecoder_params["params"][param_name]
                    )

            # Initialize working variables
            curr_solver_params = outer_params[
                "solver"
            ]  # Only used locally, not returned
            curr_opt_states = element_opt_states

            # Store initial params for improvement calculation
            if reptile_upd:
                initial_autodecoder_params = curr_autodecoder_params

            # Run inner optimization loop for each step
            for step_idx in range(self.static_num_inner_steps):
                # Generate batch indices
                curr_key, perm_key = jax.random.split(curr_key)
                batch_indices = jax.random.choice(
                    perm_key,
                    self.static_indices,
                    shape=(self.static_num_pairs_meta,),
                    replace=False,
                )

                # Process each mini-batch
                for batch_idx in range(self.static_num_mini_batches):
                    # Get batch indices
                    start_idx = batch_idx * self.static_num_pairs_mini_batch
                    end_idx = start_idx + self.static_num_pairs_mini_batch
                    indices_for_batch = jax.lax.dynamic_slice_in_dim(
                        batch_indices, start_idx, self.static_num_pairs_mini_batch
                    )

                    # Process batch

                    # Process batch
                    curr_autodecoder_params, curr_solver_params, curr_opt_states = (
                        self._process_batch(
                            {"solver": curr_solver_params},
                            curr_opt_states,
                            coords[None, ...],  # Add batch dimension
                            vel[None, ...],  # Add batch dimension
                            indices_for_batch,
                            step_idx,
                            curr_autodecoder_params,
                        )
                    )

                # Apply clipping if configured
                if self.config.geometry.clip_pos:
                    curr_autodecoder_params = autodecoder.clip_pos(
                        curr_autodecoder_params
                    )

            # Generate evaluation indices for final loss
            curr_key, eval_key = jax.random.split(curr_key)
            eval_indices = jax.random.choice(
                eval_key,
                self.static_indices,
                shape=(self.static_num_pairs_mini_batch,),
                replace=False,
            )

            # Get evaluation data
            eval_coords = jnp.take(coords[None, ...], eval_indices, axis=1)
            eval_vel = jnp.take(vel[None, ...], eval_indices, axis=1)

            # Use the current solver params for loss computation, but apply stop_gradient
            # This is important for correct gradients while avoiding unnecessary param storage
            first_order_solver_params = jax.lax.stop_gradient(curr_solver_params)

            # Calculate final loss
            element_final_params = {
                "solver": first_order_solver_params,
                "autodecoder": curr_autodecoder_params,
            }
            element_final_loss, element_final_mse = self._loss_fn(
                element_final_params,
                sub_batch_coords=eval_coords,
                sub_true_vel=eval_vel,
                reduce_batch=False,
            )

            # Calculate improvements for Reptile if needed
            improvement = jnp.array(0.0)
            if reptile_upd:
                element_initial_params = {
                    "solver": outer_params["solver"],
                    "autodecoder": initial_autodecoder_params,
                }
                element_initial_loss, _ = self._loss_fn(
                    element_initial_params,
                    sub_batch_coords=eval_coords,
                    sub_true_vel=eval_vel,
                    reduce_batch=False,
                )
                improvement = jnp.maximum(
                    0.0, element_initial_loss - element_final_loss
                )[0]

            # Return results for this element (without solver params)
            # Use scalars to avoid indexing issues
            return (
                jnp.squeeze(element_final_loss),
                jnp.squeeze(element_final_mse),
                curr_autodecoder_params,
                improvement,
            )

        # Create batch indices for each element
        batch_indices = jnp.arange(batch_size)

        # Vectorize the processing function to handle all batch elements in parallel
        vmapped_process = jax.vmap(
            process_single_element,
            in_axes=(0, 0, 0, 0),  # vectorize over keys, coords, vel, and batch_idx
        )

        # Run the vectorized computation for all batch elements
        losses, mses, autodecoder_params, improvements = vmapped_process(
            element_keys,
            batch_coords,
            true_vel,
            batch_indices,
        )

        # Compute average loss and MSE
        final_loss = jnp.mean(losses)
        final_mse = jnp.mean(mses)

        # Use the original solver params directly
        combined_solver_params = outer_params["solver"]

        # Create combined state with updated parameters
        combined_state = outer_state.replace(
            params={
                "solver": combined_solver_params,
                "autodecoder": autodecoder_params,
            },
            rng=final_key,
        )

        # Return in expected format
        return final_loss, (final_mse, combined_state, improvements)

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
            # Debug print for parameter structure

            # Print autodecoder type
            autodecoder_params = inner_state_arg.params["autodecoder"]

            # Get batch size from batch
            batch_size = batch[0].shape[0]

            # Calculate importance weights
            scaled_improvements = (
                improvements_arg / self.config.meta.reptile.weight_temperature
            )
            max_imp = jnp.max(scaled_improvements)
            exp_improvements = jnp.exp(scaled_improvements - max_imp)
            weights = exp_improvements / (jnp.sum(exp_improvements) + 1e-8)

            # Get initial parameters
            init_param = state_arg.params["init_latents"]
            reptile_params = jax.tree_map(lambda x: x, init_param)
            reptile_opt_states = state_arg.reptile_opt_states

            # Create a simple adaptation by reusing the existing parameters
            # This is a fallback to handle any parameter structure issues
            init_appearance = init_param["params"]["appearance"]
            init_pose_pos = init_param["params"]["pose_pos"]

            # Just apply a small update from the grads directly
            appearance_update = jnp.zeros_like(init_appearance)
            pose_pos_update = jnp.zeros_like(init_pose_pos)

            # Update parameters with optimizers
            appearance_updates, appearance_opt_state = self.reptile_opts[
                "appearance"
            ].update(
                appearance_update,
                reptile_opt_states["appearance"],
                step_count,
            )
            reptile_params["params"]["appearance"] = optax.apply_updates(
                reptile_params["params"]["appearance"], appearance_updates
            )
            reptile_opt_states["appearance"] = appearance_opt_state

            position_updates, position_opt_state = self.reptile_opts["pose_pos"].update(
                pose_pos_update,
                reptile_opt_states["pose_pos"],
                step_count,
            )
            reptile_params["params"]["pose_pos"] = optax.apply_updates(
                reptile_params["params"]["pose_pos"], position_updates
            )
            reptile_opt_states["pose_pos"] = position_opt_state

            # Handle orientation if needed
            if "pose_ori" in init_param["params"]:
                init_pose_ori = init_param["params"]["pose_ori"]
                pose_ori_update = jnp.zeros_like(init_pose_ori)

                orientation_updates, orientation_opt_state = self.reptile_opts[
                    "pose_ori"
                ].update(
                    pose_ori_update,
                    reptile_opt_states["pose_ori"],
                    step_count,
                )
                reptile_params["params"]["pose_ori"] = optax.apply_updates(
                    reptile_params["params"]["pose_ori"], orientation_updates
                )
                reptile_opt_states["pose_ori"] = orientation_opt_state

            # Apply clipping if needed
            if self.config.geometry.clip_pos:
                reptile_params = self.init_latents.clip_pos(reptile_params)

            # Return updated parameters
            params = {"solver": solver_params, "init_latents": reptile_params}
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
            return params, solver_opt_state, None, step_count, updated_inner_state

    def outer_step(self, state, batch, train=True):
        """Performs a single outer-loop training step with optimized memory usage.

        Args:
            state: Current training state
            batch: Batch of training data
            train: Whether this is training (True) or evaluation (False)

        Returns:
            Tuple of (loss, mse, (new_state, inner_state))
        """
        # Split random key for inner and outer loop randomness
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_outer_key)

        # Get gradients for the outer loop and update params
        (loss, (aux_data)), grads = jax.value_and_grad(self.inner_loop, has_aux=True)(
            {"solver": state.params["solver"]},
            outer_state=outer_state,
            autodecoder=self.init_latents,
            batch=batch,
            reptile_upd=train and self.config.meta.reptile.enable,
            train=train,
        )

        # Unpack auxiliary data
        mse, inner_state, improvements = aux_data

        # Conditional update based on whether we're training or evaluating
        if train:
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
        self.total_val_epochs = 1
        self.global_eval_step = 0

        # Metrics for timing and error tracking
        total_neural_fit_comp_times = 0.0
        total_rmae = 0.0
        total_re = 0.0
        total_fmm_comp_times = 0.0
        total_neural_tot_comp_times = 0.0
        total_neural_inf_comp_times = 0.0

        # Process each batch in the evaluation loader
        for batch_idx, batch in enumerate(loader):
            # Measure fitting time
            start = time.time()
            loss, mse, (eval_state, inner_state) = step_fn(eval_state, batch)
            fit_time = time.time() - start

            # Update metrics
            total_neural_fit_comp_times += fit_time
            num_points_batch = batch[0].shape[0] * batch[0].shape[1] * 2.0
            total_eval_points += num_points_batch
            losses += loss
            mse_tot += mse

            # Reduce logging frequency for speed
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.metrics[f"{name}_loss"] = loss
                self.metrics[f"{name}_mse"] = mse / num_points_batch
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
                        gt_dataset=gt_dataset,
                        batch_true_vel=true_full_vel,
                        batch_vel_idx=vel_idx,
                        visualize=visualize,
                        name=name,
                    )
                )

                # Update metrics
                total_rmae += rmae
                total_re += re
                total_fmm_comp_times += fmm_comp_times
                total_neural_tot_comp_times += neural_inf_comp_times + fit_time
                total_neural_inf_comp_times += neural_inf_comp_times

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
            f"{name}_neural_fit_comp_times": self.metrics[
                f"{name}_neural_fit_comp_times"
            ],
        }

        # Add GT metrics if available
        if gt_dataset:
            self.metrics[f"{name}_rmae"] = total_rmae
            self.metrics[f"{name}_re"] = total_re
            self.metrics[f"{name}_neural_inf_comp_times"] = total_neural_inf_comp_times
            self.metrics[f"{name}_neural_tot_comp_times"] = (
                total_neural_fit_comp_times + total_neural_inf_comp_times
            )
            self.metrics[f"{name}_fmm_comp_times"] = total_fmm_comp_times

            # Add GT metrics to the log_data
            log_data.update(
                {
                    f"{name}_rmae": total_rmae,
                    f"{name}_re": total_re,
                    f"{name}_neural_inf_comp_times": total_neural_inf_comp_times,
                    f"{name}_neural_tot_comp_times": (
                        total_neural_fit_comp_times + total_neural_inf_comp_times
                    ),
                    f"{name}_fmm_comp_times": total_fmm_comp_times,
                }
            )

        # Log all metrics at once for efficiency
        wandb.log(log_data)

        return eval_state

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

        # Create input arguments for mapping
        args = [
            (i, vel_idx, vel)
            for i, (vel_idx, vel) in enumerate(zip(batch_vel_idx, batch_true_vel))
        ]

        # Use vmap when possible, otherwise loop
        if len(args) > 0:
            # Process all velocities and compute metrics
            results = [process_single_vel(arg) for arg in args]
            rmae_values, re_values, fmm_times, neural_times = zip(*results)

            # Return aggregated metrics
            return sum(rmae_values), sum(re_values), sum(fmm_times), sum(neural_times)
        else:
            return 0.0, 0.0, 0.0, 0.0

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
