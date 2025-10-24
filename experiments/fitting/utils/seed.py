import numpy as np
import random
import torch
import pytorch_lightning as pl


def set_global_determinism(seed):
    """
    Set deterministic behavior for all libraries and configure environment
    to ensure reproducible results across runs.

    Args:
        seed: Integer seed for deterministic behavior
        disable_gpu: If True, forces operations to run on CPU for better determinism
    """

    # Set all standard library random seeds
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Additional PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use PyTorch Lightning's seed_everything as well (covers more bases)
    pl.seed_everything(
        seed, workers=True
    )  # Add workers=True to make dataloader workers deterministic

    print(f"Global seed set to {seed}")

    return seed
