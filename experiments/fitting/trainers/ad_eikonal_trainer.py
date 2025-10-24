import jax.numpy as jnp
import numpy as np
import optax
import jax
from tqdm import tqdm
from experiments.fitting.trainers._base.base_eikonal_trainer import BaseEikonalTrainer
import time

from experiments.fitting.utils.logging import logger  # Add this line
from flax import struct


class AutoDecodingEikonalTrainer(BaseEikonalTrainer):
    class TrainState(BaseEikonalTrainer.TrainState):
        autodecoder_opt_state: optax.OptState = struct.field(pytree_node=True)
        autodecoder_steps: int = -1
        vmin: float = 0.0
        vmax: float = 1.0

    def __init__(
        self,
        config,
        solver,
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

        self.train_autodecoder = train_autodecoder
        self.val_autodecoder = val_autodecoder
        self.test_autodecoder = test_autodecoder

        self._setup_optimizers()

    def _setup_optimizers(self):
        self.solver_opt = optax.chain(
            optax.clip_by_global_norm(self.config.training.gradient_clip_val),
            optax.adamw(
                learning_rate=self.config.optimizer.learning_rate_solver,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                weight_decay=self.config.optimizer.weight_decay,
            ),
        )

        # Create optimizers for different autodecoder parts with gradient clipping
        autodecoder_codes_opt = optax.chain(
            optax.clip_by_global_norm(self.config.training.gradient_clip_val),
            optax.adamw(
                learning_rate=self.config.optimizer.learning_rate_codes,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                weight_decay=self.config.optimizer.weight_decay,
            ),
        )

        autodecoder_poses_opt = optax.chain(
            optax.clip_by_global_norm(self.config.training.gradient_clip_val),
            optax.adamw(
                learning_rate=self.config.optimizer.learning_rate_poses,
                b1=0.9,
                b2=0.999,
                eps=1e-8,
                weight_decay=self.config.optimizer.weight_decay,
            ),
        )

        # Define a function to partition the parameters
        def param_labels(params):
            labels = {}
            for key in params["params"]:
                if key == "appearance":
                    labels["params"] = {"appearance": "appearance"}
                elif key in ["pose_pos", "pose_ori"]:
                    if "params" not in labels:
                        labels["params"] = {}
                    labels["params"][key] = "pose"

            return labels

        # Create a dictionary mapping parameter groups to optimizers
        optimizer_dict = {
            "appearance": autodecoder_codes_opt,
            "pose": autodecoder_poses_opt,
        }

        self.autodecoder_opt = optax.multi_transform(optimizer_dict, param_labels)

    def init_train_state(self):
        """Initializes the training state with proper gradient clipping.

        Returns:
            TrainState: The training state.
        """
        # Initialize optimizer and scheduler with gradient clipping

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, solver_key = jax.random.split(key)
        key, autodecoder_key = jax.random.split(key)

        # Create a test batch to get the shape of the latent space
        autodecoder_params = self.train_autodecoder.init(
            autodecoder_key,
            jnp.ones(self.config.data.train_batch_size, dtype=jnp.int32),
        )

        dim_signal = self.config.geometry.dim_signal
        if self.config.geometry.input_space == "Spherical":
            dim_signal -= 1

        # Initialize solver
        sample_coords = jax.random.normal(
            solver_key,
            (
                self.config.data.train_batch_size,
                self.config.data.num_pairs,
                2,
                dim_signal,
            ),
        )

        # autodecoder_params = cast_params_to_high_precision(autodecoder_params)
        p, a = self.train_autodecoder.apply(
            autodecoder_params,
            jnp.ones(self.config.data.train_batch_size, dtype=jnp.int32),
        )

        solver_params = self.solver.init(
            solver_key,
            sample_coords,
            p,
            a,
        )

        train_state = self.TrainState(
            params={"solver": solver_params, "autodecoder": autodecoder_params},
            solver_opt_state=self.solver_opt.init(solver_params),
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            rng=key,
            autodecoder_steps=-1,
            vmin=self.solver.vmin,
            vmax=self.solver.vmax,
        )

        return train_state

    def create_functions(self):
        """Create training functions."""
        super().create_functions()

        self.train_step = jax.jit(
            jax.tree_util.Partial(
                self.step,
                autodecoder=self.train_autodecoder,
                train=True,
            )
        )

        self.val_step_auto = jax.jit(
            jax.tree_util.Partial(
                self.step,
                autodecoder=self.val_autodecoder,
                train=False,
            )
        )

        self.val_step_inf = jax.jit(
            jax.tree_util.Partial(
                self.inf_step,
                autodecoder=self.val_autodecoder,
            )
        )

        self.test_step_auto = jax.jit(
            jax.tree_util.Partial(
                self.step,
                autodecoder=self.test_autodecoder,
                train=False,
            )
        )

        self.test_step_inf = jax.jit(
            jax.tree_util.Partial(
                self.inf_step,
                autodecoder=self.test_autodecoder,
            )
        )

        self.validate_epoch = lambda state: self.eval_epoch(state, name="val")
        self.test_epoch = lambda state: self.eval_epoch(state, name="test")

    def inf_step(self, state, batch, autodecoder):
        true_vel, batch_coords, vel_idx = batch
        p, a = autodecoder.apply(state.params["autodecoder"], vel_idx)

        outputs = self.solver.apply(
            state.params["solver"],
            method=self.solver.times_grad_vel,
            inputs=batch_coords,
            p=p,
            a=a,
            aux_vel=True,
        )

        pred_times, input_grads, norm_input_grad, pred_vel = outputs
        loss = self.eiko_loss(norm_input_grad, true_vel, abs_loss=True)
        mse = jnp.sum((pred_vel - true_vel) ** 2)

        return (
            jnp.asarray(loss, dtype=true_vel.dtype),
            jnp.asarray(mse, dtype=true_vel.dtype),
            state,
        )

    def step(self, state, batch, autodecoder, train=True):
        """Performs a single training step with optimized JAX operations.

        Args:
            state: Current training state
            batch: Current data batch (true_vel, batch_coords, vel_idx)
            autodecoder: Autodecoder model
            train: Whether to update solver params (static argument for JIT)

        Returns:
            Tuple of (loss, mse, updated_state)
        """
        # Unpack batch
        true_vel, batch_coords, vel_idx = batch

        # Define loss function
        def loss_fn(params):
            p, a = autodecoder.apply(params["autodecoder"], vel_idx)
            outputs = self.solver.apply(
                params["solver"],
                method=self.solver.times_grad_vel,
                inputs=batch_coords,
                p=p,
                a=a,
                aux_vel=True,
            )
            pred_times, input_grads, norm_input_grad, pred_vel = outputs
            loss = self.eiko_loss(norm_input_grad, true_vel, abs_loss=True)
            mse = jnp.sum((pred_vel - true_vel) ** 2)

            return loss, mse

        # Compute gradients
        (loss, mse), param_grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )

        # Define update functions for training and evaluation modes
        if train:
            solver_updates, solver_opt_state = self.solver_opt.update(
                param_grads["solver"],
                state.solver_opt_state,
                params=state.params["solver"],  # Add this line
            )
            solver_params = optax.apply_updates(state.params["solver"], solver_updates)
        else:
            solver_params, solver_opt_state = (
                state.params["solver"],
                state.solver_opt_state,
            )

        # Always update autodecoder
        autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(
            param_grads["autodecoder"],
            state.autodecoder_opt_state,
            params=state.params["autodecoder"],  # Add this line
        )
        autodecoder_params = optax.apply_updates(
            state.params["autodecoder"], autodecoder_updates
        )

        # Conditionally clip positions
        autodecoder_params = jax.lax.cond(
            self.config.geometry.clip_pos,
            lambda p: autodecoder.clip_pos(p, vel_idx),
            lambda p: p,
            autodecoder_params,
        )

        # Update state with new parameters
        updated_state = state.replace(
            params={"solver": solver_params, "autodecoder": autodecoder_params},
            solver_opt_state=solver_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
        )

        return (
            jnp.asarray(loss, dtype=true_vel.dtype),
            jnp.asarray(mse, dtype=true_vel.dtype),
            updated_state,
        )

    def fit_autodecoding(self, state, name="val"):
        if name == "val":
            autodecoder = self.val_autodecoder
            loader = self.val_loader
            step_fn = self.val_step_auto
        else:
            autodecoder = self.test_autodecoder
            loader = self.test_loader
            step_fn = self.test_step_auto

        key, init_key = jax.random.split(state.rng)
        autodecoder_params = autodecoder.init(
            init_key, jnp.ones(self.config.data.test_batch_size, dtype=jnp.int32)
        )

        # Create validation state
        eval_state = state.replace(
            params={
                "solver": state.params["solver"],
                "autodecoder": autodecoder_params,
            },
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            rng=key,
        )

        if name == "val":
            if self.config.test.num_epochs_auto == -1:
                self.total_eval_epochs = (
                    max(self.epoch, self.config.test.min_num_epochs) + 1
                )
            else:
                self.total_eval_epochs = self.config.test.num_epochs_auto + 1

            eval_state = eval_state.replace(autodecoder_steps=self.total_eval_epochs)
        else:
            self.total_eval_epochs = eval_state.autodecoder_steps

        self.global_eval_step = 0
        neural_fit_comp_times = 0.0
        # Loop over batches
        for epoch in range(1, self.total_eval_epochs):
            losses = 0
            mse_tot = 0
            total_eval_points = 0
            self.cur_eval_epoch = epoch

            for batch_idx, batch in enumerate(loader):
                # if epoch == 1 and batch_idx == 0:
                #     loss_aux, _, _ = step_fn(eval_state, batch)
                #     loss_aux.block_until_ready()
                #     loss_np_aux = float(jax.device_get(loss_aux))

                start = time.time()
                loss, mse, eval_state = step_fn(eval_state, batch)
                loss_np = float(jax.device_get(loss))
                neural_fit_comp_times += time.time() - start
                num_points_batch = batch[0].shape[0] * batch[0].shape[1] * 2.0
                total_eval_points += num_points_batch

                mse_np = float(jax.device_get(mse))

                losses += loss_np
                mse_tot += mse_np

                # Log every n steps
                if batch_idx % self.config.logging.log_every_n_steps == 0:
                    self.update_prog_bar(step=batch_idx, train=False)

                if (
                    self.visualize_reconstruction is not None
                    and not self.config.logging.debug
                    and self.global_eval_step
                    % self.config.logging.visualize_every_n_steps
                    == 0
                ):
                    eval_state = self.visualize_batch(
                        eval_state,
                        batch,
                        name=f"{name}/recon-fitting",
                        base_dataset=loader.dataset.base_dataset,
                        autodecoder=autodecoder,
                    )

                # Increment global val step
                self.global_eval_step += 1

        self.metrics[f"{name}_neural_fit_comp_times"] = neural_fit_comp_times

        self.cur_eval_epoch = 0

        return eval_state

    def eval_epoch(self, state, name="val"):
        """Validates the model. Since we're doing autodecoding, requires
            training a validation autodecoder from scratch.

        Args:
            state: The current training state.
        Returns:
            state: The updated training state.
        """
        # Initialize autodecoder

        if name == "val":
            autodecoder = self.val_autodecoder
            loader = self.val_loader
            gt_dataset = self.gt_dataset_val
            step_fn = self.val_step_inf
        else:
            autodecoder = self.test_autodecoder
            loader = self.test_loader
            gt_dataset = self.gt_dataset_test
            step_fn = self.test_step_inf

        # Create validation state
        eval_state = self.fit_autodecoding(state, name=name)
        state = state.replace(autodecoder_steps=eval_state.autodecoder_steps)
        total_eval_points = 0
        losses = 0
        mse_tot = 0

        # NEW: Critical - we'll use one consistent step for all validation logs
        # This should be the current global step from training
        validation_log_step = self.global_step

        # Set up local counter for batches
        local_batch_idx = 0
        self.global_eval_step = 0

        for batch_idx, batch in enumerate(loader):
            loss, mse, eval_state = step_fn(eval_state, batch)
            num_points_batch = batch[0].shape[0] * batch[0].shape[1] * 2.0
            total_eval_points += num_points_batch

            losses += loss
            mse_tot += mse

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                loss_np = float(jax.device_get(loss))
                mse_np = float(jax.device_get(mse))

                logger.add_to_buffer(
                    {f"{name}_loss": loss_np, f"{name}_mse": mse_np / num_points_batch},
                    commit=False,
                )
                self.update_prog_bar(step=batch_idx, train=False)

            # Increment global val step
            self.global_eval_step += 1

        # Reset val epoch
        self.total_eval_epochs = 0
        self.global_eval_step = 0

        # Visualize last batch
        if self.visualize_reconstruction is not None:
            eval_state = self.visualize_batch(
                eval_state,
                batch,
                base_dataset=loader.dataset.base_dataset,
                name=f"{name}/recon-final",
                autodecoder=autodecoder,
            )

        # Update epoch loss by last loss
        self.metrics[f"{name}_eiko_epoch"] = losses / len(loader)
        self.metrics[f"{name}_mse_epoch"] = mse_tot / total_eval_points

        log_data = {
            f"{name}_eiko_epoch": self.metrics[f"{name}_eiko_epoch"],
            f"{name}_mse_epoch": self.metrics[f"{name}_mse_epoch"],
            f"{name}_neural_fit_comp_times": self.metrics[
                f"{name}_neural_fit_comp_times"
            ],
        }

        print("Started validating against GT")
        if gt_dataset:
            for ds in gt_dataset:
                print(f"Validating against {ds }")
                name_ds = ds.name
                rmae, re, fmm_comp_times, neural_inf_comp_times = (
                    self.validate_against_gt(
                        dataset=loader.dataset.base_dataset,
                        name=name,
                        autodecoder=autodecoder,
                        state=eval_state,
                        gt_dataset=ds,
                    )
                )
                self.metrics[f"{name}_{name_ds}_re"] = re

                log_data.update(
                    {
                        f"{name}_{name_ds}_rmae": rmae,
                        f"{name}_{name_ds}_re": re,
                        f"{name}_{name_ds}_neural_inf_comp_times": neural_inf_comp_times,
                        f"{name}_{name_ds}_neural_tot_comp_times": (
                            self.metrics[f"{name}_neural_fit_comp_times"]
                            + neural_inf_comp_times
                        ),
                        f"{name}_{name_ds}_fmm_comp_times": fmm_comp_times,
                    }
                )

        if name == "val":
            if "val_top_re" in self.metrics:
                self.cur_val_metric = self.metrics["val_top_re"]
            elif "val_full_re" in self.metrics:
                self.cur_val_metric = self.metrics["val_full_re"]
            else:
                # self.cur_val_metric = self.metrics["val_eiko_epoch"]
                self.cur_val_metric = self.metrics["train_eiko_epoch"]

        # Log all metrics at once with explicit commit
        logger.log(log_data, step=validation_log_step, commit=True)

        return state, eval_state

    def validate_against_gt(self, dataset, name, autodecoder, state, gt_dataset):
        fmm_comp_times = 0.0
        neural_inf_comp_times = 0.0
        rmae = 0.0
        re = 0.0

        num_vel = len(dataset)

        # Warm-up call (to compile the jitted function _all_coords_no_plot)
        coords = gt_dataset.grid_data["coords"]
        p, a = autodecoder.apply(state.params["autodecoder"], jnp.array([0]))
        pred_times_all = self._all_coords_no_plot(state.params["solver"], coords, p, a)
        pred_times_all.block_until_ready()

        # Enable gradient updates
        for vel, vel_idx in tqdm(dataset, "Ground truth val"):

            # Get the latent codes, repeat for each coordinate.

            p, a = autodecoder.apply(
                state.params["autodecoder"], jnp.array([vel_idx], dtype=jnp.int32)
            )

            visualize = (
                self.visualize_gt is not None
                and vel_idx >= num_vel - self.config.eikonal.ground_truth.num_visualized
            )

            rmae_aux, re_aux, fmm_comp_times_aux, neural_inf_comp_times_aux = (
                self._validate_one_vel_against_gt(
                    vel,
                    vel_idx,
                    p,
                    a,
                    gt_dataset=gt_dataset,
                    solver_params=state.params["solver"],
                    visualize=visualize,
                    name=name,
                )
            )

            rmae += rmae_aux
            re += re_aux
            fmm_comp_times += fmm_comp_times_aux
            neural_inf_comp_times += neural_inf_comp_times_aux

        rmae = rmae / num_vel

        re = re / num_vel

        return rmae, re, fmm_comp_times, neural_inf_comp_times

    def visualize_batch(self, state, batch, name, base_dataset, autodecoder=None):
        """Visualizes the current batch.

        Args:
            state: The current training state.
            batch: The current batch.
            name: The name of the visualization.
            train: Whether we are training or validating.
        """

        if self.visualize_reconstruction is not None:
            true_vel, batch_coords, vel_idx = batch
            true_full_vel = np.stack([base_dataset[idx][0] for idx in vel_idx])

            if autodecoder is None:
                autodecoder = self.train_autodecoder

            p, a = autodecoder.apply(state.params["autodecoder"], vel_idx)

            rng, rng_plot = jax.random.split(state.rng)

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

            state = state.replace(rng=rng)

        return state
