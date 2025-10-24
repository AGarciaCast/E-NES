import os
import numpy as np
from torch.utils.data import Dataset


class OpenFWIDataset(Dataset):
    """OpenFWI  dataset
    For convenience, in this class, a batch refers to a npy file
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        file_size: # of samples in each npy file
        transform: transformation applied to data
        create_3d: 
    """

    def __init__(self, anno, file_size=500, create_3d=None, transform=None, preload=False):

        if not os.path.exists(anno):
            print(f"Annotation file {anno} does not exists")

        self.file_size = file_size
        self.transform = transform
        self.create_3d = create_3d
        with open(anno, "r") as f:
            self.batches = f.read().splitlines()

    # Load from one line
    def load_every(self, data_path):
        data = np.load("./experiments/fitting/datasets/euclidean" + data_path)
        data = data.astype("float32")
        return data

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size

        data = self.load_every(self.batches[batch_idx])
        data = data[sample_idx]

        # Apply data transformations
        if self.transform:
            data = self.transform(data)

        # Add third spatial dimension if needed by copying the data along the third dimension
        if self.create_3d:
            orig_shape = data.shape
            data = np.tile(data, (orig_shape[-1], 1, 1)) # Adding third dimension to x-axis to still keep the sand layers
                                                          # aligned with the horizontal axis.

        return data, np.array([])

    def __len__(self):
        return len(self.batches) * self.file_size                                  
