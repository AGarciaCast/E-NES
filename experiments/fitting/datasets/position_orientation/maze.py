import numpy as np
from torch.utils.data import Dataset

import numpy as np

from experiments.fitting.datasets.position_orientation.utils import (
    apply_3d_gaussian_blur,
)


class MazeData(Dataset):
    def __init__(
        self, train=True, num_ori=32, vmin=1, vmax=10, wind=False, gaussian=True, seed=0
    ):
        assert vmin >= 1

        if train:
            # Load the training maps
            maps_array = np.load(
                "./experiments/fitting/datasets/position_orientation/train_maps.npy"
            )
        else:
            # Load the testing maps
            maps_array = np.load(
                "./experiments/fitting/datasets/position_orientation/test_maps.npy"
            )

        maps_array = 1 - maps_array.astype(np.float32)
        # Tile the maps for different orientation calculations
        maps_tiled = np.tile(
            maps_array[..., None], (1, 1, 1, num_ori)
        )  # Shape: (200, 50, 50, num_ori)

        data = maps_tiled

        if gaussian:

            # Apply Gaussian blur to the data
            # data = binary_dilation(data)
            # data = apply_3d_gaussian_blur(data, sigma=3)
            # data = apply_3d_gaussian_blur(data * maps_tiled, sigma=1)
            # data = apply_3d_gaussian_blur(data * maps_tiled, sigma=2)
            # data = (maps_tiled - 1 + data) * data
            # data = maps_tiled * 0.2 + data * 0.8

            data = apply_3d_gaussian_blur(data, sigma=3)
            data = apply_3d_gaussian_blur(data * maps_tiled, sigma=1)
            data = apply_3d_gaussian_blur(data * maps_tiled, sigma=2)
            data = maps_tiled * 0.1 + data * 0.9

        data = data * 100

        if wind:
            # Generate random orientations for each map
            theta = np.linspace(
                0, 2 * np.pi, num_ori, endpoint=False
            )  # Shape: (num_ori,)
            np.random.seed(seed)
            ori = np.random.uniform(
                0, 2 * np.pi, size=(len(maps_array))
            )  # Shape: (200,) - removed extra dimension

            # Reshape ori for proper broadcasting
            ori_reshaped = ori[:, None, None, None]  # Shape: (200, 1, 1, 1)

            speed = (
                2
                - np.cos(
                    2
                    * (
                        theta[None, None, None, ...]  # Shape: (1, 1, 1, num_ori)
                        - ori_reshaped  # Shape: (200, 1, 1, 1)
                    )
                )
                - 0.4 * np.cos(3 * (theta[None, None, None, ...] - ori_reshaped))
            )
            # Apply the wind speed to the data
            data = data * speed  # Should maintain shape (200, 50, 50, num_ori)

        data = data * maps_tiled

        # Normalize the data between vmin and vmax
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = data * (vmax - vmin) + vmin
        # Clip the data to ensure it falls within the range [vmin, vmax]
        data = np.clip(data, vmin, vmax)

        self.data = data
        self.num_samples = len(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], np.array([])


if __name__ == "__main__":

    # Create the dataset
    dataset = MazeData(wind=True, gaussian=True, vmax=100, train=False)
    print(f"Dataset length: {len(dataset)}")
    maze = dataset[0][0]

    min_maze = np.min(maze, axis=-1)
    # Visualize the first map
    import matplotlib.pyplot as plt

    plt.imshow(min_maze)
    plt.title("Maze Map with Wind and Gaussian Blur")
    # save the figure
    plt.savefig("maze_map.png")
    plt.close()
    speed = np.max(np.max(maze, axis=0), axis=0)
    ori = np.linspace(0, 2 * np.pi, len(speed), endpoint=False)
    plt.figure(figsize=(10, 10))
    plt.title("Wind Speed and Orientation")
    plt.scatter(0, 0)

    # Create the polar coordinates
    x = speed * np.cos(ori)
    y = speed * np.sin(ori)

    # Add the first point (at orientation 0) again at the end (orientation 2π)
    # This effectively adds speed[0] at position 2π
    x = np.append(x, speed[0] * np.cos(0))
    y = np.append(y, speed[0] * np.sin(0))

    plt.plot(x, y)
    plt.axis("equal")  # Make the plot have equal scaling
    # plt.grid(True)
    plt.savefig("maze_map_speed.png")
