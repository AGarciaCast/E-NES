import numpy as np
from torch.utils.data import Dataset


import numpy as np
from experiments.fitting.datasets.position_orientation.utils import (
    apply_3d_gaussian_blur,
)


class ObstaclesData(Dataset):
    def __init__(
        self,
        num_samples=150,
        size=70,
        num_circles=3,
        radius=12,
        num_ori=32,
        vmin=1,
        vmax=10,
        wind=False,
        gaussian=True,
        seed=0,
    ):
        # assert vmin >= 1
        np.random.seed(seed)

        # maps_array = np.empty((num_samples, size, size), dtype=np.float32)

        # for i in range(num_samples):
        #     image = np.ones((size, size), dtype=np.float32)

        #     # Generate random centers for the circles
        #     # Ensure circles are fully within the image boundaries
        #     centers = []
        #     for _ in range(num_circles):
        #         x = np.random.randint(radius, size - radius)
        #         y = np.random.randint(radius, size - radius)
        #         centers.append((x, y))

        #         # Create a circle at the random location
        #         y_indices, x_indices = np.ogrid[:size, :size]
        #         dist_from_center = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
        #         mask = dist_from_center < radius
        #         image[mask] = 0.0

        #     maps_array[i] = image

        maps_array = np.empty((num_samples, size, size), dtype=np.float32)

        for i in range(num_samples):
            image = np.ones((size, size), dtype=np.float32)

            # Generate random centers for the circles with collision detection
            centers = []
            attempts = 0
            max_attempts = 1000  # Prevent infinite loops

            while len(centers) < num_circles and attempts < max_attempts:
                # Generate a potential new circle
                x = np.random.randint(radius, size - radius)
                y = np.random.randint(radius, size - radius)

                # Check if this circle overlaps with any existing circles
                valid_position = True
                for cx, cy in centers:
                    # Calculate distance between centers
                    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    # If centers are closer than 2*radius, circles would overlap
                    if distance < 2 * radius:
                        valid_position = False
                        break

                if valid_position:
                    centers.append((x, y))

                attempts += 1

            # If we couldn't place all circles without overlap
            if len(centers) < num_circles:
                print(
                    f"Warning: Could only place {len(centers)} circles without overlap in sample {i}"
                )

            # Draw all the non-overlapping circles
            for x, y in centers:
                y_indices, x_indices = np.ogrid[:size, :size]
                dist_from_center = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
                mask = dist_from_center < radius
                image[mask] = 0.0

            maps_array[i] = image

        # Tile the maps for different orientation calculations
        maps_tiled = np.tile(
            maps_array[..., None], (1, 1, 1, num_ori)
        )  # Shape: (200, 50, 50, num_ori)

        data = maps_tiled

        if gaussian:

            # Apply Gaussian blur to the data
            data = apply_3d_gaussian_blur(data, sigma=5, constant_values=1)
            # data = apply_3d_gaussian_blur(data * maps_tiled, sigma=1, constant_values=1)
            data = apply_3d_gaussian_blur(data * maps_tiled, sigma=2, constant_values=1)
            data = (maps_tiled - 1 + data) * data
            data = maps_tiled * 0.2 + data * 0.8

            # data = apply_3d_gaussian_blur(data, sigma=3, constant_values=1)
            # data = apply_3d_gaussian_blur(data * maps_tiled, sigma=1, constant_values=1)
            # data = apply_3d_gaussian_blur(data * maps_tiled, sigma=2, constant_values=1)
            # data = maps_tiled * 0.1 + data * 0.9

        # data = data * 100

        if wind:
            # Generate random orientations for each map
            theta = np.linspace(
                0, 2 * np.pi, num_ori, endpoint=False
            )  # Shape: (num_ori,)

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

        # data = data * maps_tiled

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
    dataset = ObstaclesData(
        num_samples=150,
        size=70,
        num_circles=3,
        radius=10,
        wind=True,
        gaussian=True,
        vmax=100,
    )
    print(f"Dataset length: {len(dataset)}")
    maze = dataset[10][0]

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
