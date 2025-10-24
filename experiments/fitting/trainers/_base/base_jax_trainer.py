from experiments.fitting.utils.logging import logger  # Add this line
from typing import Any

from tqdm.auto import tqdm

# For trainstate
from flax import struct, core
import optax
import jax.numpy as jnp

# Checkpointing
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

import os


class JaxTrainer:

    class TrainState(struct.PyTreeNode):
        params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
        time_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
        opt_state: optax.OptState = struct.field(pytree_node=True)
        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(
        self,
        config,
        num_epochs,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        seed=42,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.seed = seed
        self.total_train_epochs = num_epochs

        # Placeholders for train and val steps
        self.train_step = None
        self.val_step = None
        self.test_step = None

        self.test_epoch = lambda state: self.val_epoch(state)

        # Keep track of training state
        self.global_step = 0
        self.epoch = 0

        # Keep track of state of validation
        self.global_val_step = 0

        # Keep track of metrics
        self.metrics = {}
        self.top_val_metric = jnp.inf
        self.cur_val_metric = jnp.inf

        # Description strings for train and val progress bars
        self.prog_bar_desc = """{state} :: epoch - {epoch}/{total_epochs} ::"""
        self.prog_bar = tqdm(
            desc=self.prog_bar_desc.format(
                state="Training",
                epoch=self.epoch,
                total_epochs=self.total_train_epochs,
            ),
            total=len(self.train_loader),
        )

        # Set checkpoint options
        if self.config.logging.checkpoint:
            checkpoint_options = ocp.CheckpointManagerOptions(
                save_interval_steps=1,
                max_to_keep=1,
            )

            # Get the wandb run directory path
            wandb_run_id = None
            if logger.use_wandb and logger.initialized:
                wandb = logger._safe_wandb_import()
                if wandb and wandb.run:
                    wandb_run_id = wandb.run.id

            # Set checkpoint directory to wandb run directory or fallback to default
            checkpoint_dir = f"./checkpoints/{wandb_run_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Ensure the path is absolute
            checkpoint_dir = os.path.abspath(checkpoint_dir)
            print(checkpoint_dir)

            self.checkpoint_manager = ocp.CheckpointManager(
                directory=checkpoint_dir,
                options=checkpoint_options,
                item_handlers={
                    "state": ocp.StandardCheckpointHandler(),
                    "config": ocp.JsonCheckpointHandler(),
                },
                item_names=["state", "config"],
            )

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            state: The training state.
        """
        raise NotImplementedError("init_train_state method must be implemented.")

    def create_functions(self):
        """Creates the functions for training and validation. Should implement train_step and val_step."""
        raise NotImplementedError("create_functions method must be implemented.")

    def train_model(self, state=None):
        """Trains the model for the given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train for.

        Returns:
            state: The final training state.
        """

        # Keep track of global step
        self.global_step = 0

        self.epoch = 0

        logger.last_logged_step = self.global_step - 1

        if state is None:
            state = self.init_train_state()

        # Log initial configuration with explicit step 0
        logger.log({"epoch": 0}, step=0)

        for epoch in range(1, self.total_train_epochs + 1):
            self.epoch = epoch
            # Log epoch start - don't commit yet to batch with epoch results
            logger.add_to_buffer({"epoch": epoch}, commit=False)

            state = self.train_epoch(state)

            # Validate every n epochs
            if epoch % self.config.test.val_every_n_epochs == 0:
                state, val_state = self.validate_epoch(state)

                self.save_checkpoint(state)

            # Explicitly flush buffer in case train_epoch or validate_epoch didn't commit
            logger.flush_buffer(step=self.global_step)

        if self.test_loader is not None:
            if self.config.logging.checkpoint:

                state = self.load_checkpoint()

            state, test_state = self.test_epoch(state)

        # Explicitly flush buffer in case train_epoch or validate_epoch didn't commit
        logger.flush_buffer()

        return state

    def train_epoch(self, state):
        """Train the model for one epoch.

        Args:
            state: The current training state.
            epoch: The current epoch.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss, state = self.train_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                logger.log({"train_loss_step": loss}, step=self.global_step)

                self.update_prog_bar(step=batch_idx)

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.metrics["train_loss_epoch"] = losses / len(self.train_loader)
        logger.log({"train_loss_epoch": self.metrics["train_loss_epoch"]})
        return state

    def eval_epoch(self, state, name="val"):
        """Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss, _ = self.val_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                logger.log({f"{name}_loss_step": loss})
                self.update_prog_bar(step=batch_idx, train=False)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.metrics[f"{name}_loss_epoch"] = losses / len(self.val_loader)
        self.cur_val_metric = self.metrics[f"{name}_loss_epoch"]
        logger.log({f"{name}_loss_epoch": self.metrics[f"{name}_loss_epoch"]})

        return state, state

    def save_checkpoint(self, state):
        """Save the current state to a checkpoint

        Args:
            state: The current training state.
        """
        if self.config.logging.checkpoint and self.top_val_metric > self.cur_val_metric:
            self.checkpoint_manager.save(
                step=self.epoch,
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    config=ocp.args.JsonSave(OmegaConf.to_container(self.config)),
                ),
            )

            self.top_val_metric = self.cur_val_metric

    def load_checkpoint(self):
        """Load the latest checkpoint"""
        ckpt = self.checkpoint_manager.restore(self.checkpoint_manager.latest_step())
        return self.TrainState(**ckpt.state)

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

        epoch = self.epoch
        total_epochs = self.total_train_epochs

        if train:
            global_step = self.global_step
        else:
            global_step = self.global_val_step

        # Update description string
        prog_bar_str = self.prog_bar_desc.format(
            state="Training" if train else "Validation",
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            global_step=global_step,
        )

        # Append metrics to description string
        if self.metrics:
            for key, value in self.metrics.items():
                prog_bar_str += f" -- {key} {value:.4f}"

        self.prog_bar.set_description_str(prog_bar_str)
