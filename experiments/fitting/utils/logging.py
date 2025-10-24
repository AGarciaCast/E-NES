import traceback
from typing import Dict, Any, Optional
import os


class WandbLogger:
    """Wandb logger with PyTorch Lightning compatibility."""

    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb
        self.initialized = False
        self.current_step = 0  # Keep track of the last step we logged to
        self.metrics_buffer = {}
        self.buffer_commit = True

        # NEW: Keep track of the last step explicitly used for logging
        self.last_logged_step = 0

        # Check environment variables for wandb control
        if os.environ.get("DISABLE_WANDB", "").lower() in ("true", "1", "t"):
            self.use_wandb = False
            print("WandbLogger: Disabled via DISABLE_WANDB environment variable")

        if not self.use_wandb:
            print("WandbLogger: Wandb logging disabled")

    def _safe_wandb_import(self):
        """Safely import wandb module."""
        try:
            import wandb

            return wandb
        except ImportError:
            print("WARNING: wandb module not available")
            self.use_wandb = False
            return None

    def init(self, **kwargs):
        """Initialize wandb with error handling."""
        if not self.use_wandb:
            return

        try:
            wandb = self._safe_wandb_import()
            if not wandb:
                return

            if not wandb.run:
                wandb.init(**kwargs)
                self.initialized = True
                print(
                    f"WandbLogger: Successfully initialized run '{kwargs.get('name', 'unnamed')}'"
                )

                # Define x-axis explicitly for Lightning compatibility
                if wandb.run is not None:
                    wandb.define_metric("trainer/global_step")
                    wandb.define_metric("*", step_metric="trainer/global_step")
        except Exception as e:
            print(f"ERROR initializing wandb: {e}")
            traceback.print_exc()
            self.use_wandb = False

    def add_to_buffer(self, metrics: Dict[str, Any], commit: bool = None):
        """Add metrics to buffer for batched logging."""
        if not metrics:
            return

        self.metrics_buffer.update(metrics)
        if commit is not None:
            self.buffer_commit = commit

    def flush_buffer(self, step: Optional[int] = None, commit: Optional[bool] = None):
        """Flush the metrics buffer to wandb."""
        if not self.metrics_buffer:
            return

        if commit is None:
            commit = self.buffer_commit

        self.log(self.metrics_buffer, step=step, commit=commit)
        self.metrics_buffer = {}
        self.buffer_commit = True

    def log(
        self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True
    ):
        """Log metrics with Lightning compatibility."""
        if not self.use_wandb or not self.initialized:
            return

        try:
            wandb = self._safe_wandb_import()
            if not wandb:
                return

            if step is None:
                # If no step is provided, use the next step after the last logged step
                step = self.last_logged_step + 1

            # NEW: Crucial update - ensure steps never go backward
            step = max(step, self.last_logged_step + 1)

            # Add the Lightning-style step field to all metrics
            metrics_with_step = dict(metrics)
            metrics_with_step["trainer/global_step"] = step

            wandb.log(metrics_with_step, step=step, commit=commit)

            # Update the last logged step
            self.last_logged_step = step

            if commit:
                self.current_step = step + 1
        except Exception as e:
            print(f"WARNING: wandb logging failed: {e}")
            traceback.print_exc()

    def log_image(
        self, name: str, image, step: Optional[int] = None, commit: bool = False
    ):
        """Log an image with proper error handling."""
        if not self.use_wandb or not self.initialized:
            return

        try:
            wandb = self._safe_wandb_import()
            if not wandb:
                return

            self.log({name: wandb.Image(image)}, step=step, commit=commit)
        except Exception as e:
            print(f"WARNING: wandb image logging failed: {e}")
            traceback.print_exc()

    def finish(self):
        """Finish the wandb run properly."""
        if not self.use_wandb or not self.initialized:
            return

        try:
            wandb = self._safe_wandb_import()
            if not wandb:
                return

            # Flush any remaining metrics
            self.flush_buffer()
            wandb.finish()
            print("WandbLogger: Successfully finished wandb run")
            self.initialized = False
        except Exception as e:
            print(f"WARNING: wandb finish failed: {e}")
            traceback.print_exc()


# Global logger instance
logger = WandbLogger(use_wandb=True)


# Convenience functions that delegate to the global logger
def init(**kwargs):
    return logger.init(**kwargs)


def log(metrics, step=None, commit=True):
    return logger.log(metrics, step, commit)


def log_image(name, image, step=None, commit=False):
    return logger.log_image(name, image, step, commit)


def finish():
    return logger.finish()
