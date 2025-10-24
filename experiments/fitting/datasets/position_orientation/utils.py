import numpy as np
from scipy.ndimage import gaussian_filter


def apply_3d_gaussian_blur(data, sigma, constant_values=0):
    """
    Apply 3D Gaussian blur to each map in the data.
    - Circular (cyclic) handling in the orientation dimension
    - Padding of 1 in the spatial dimensions

    Parameters:
    data : numpy array with shape (batch, height, width, orientations)
    sigma : standard deviation for Gaussian kernel

    Returns:
    blurred_data : numpy array with same shape as data
    """
    # Get shape of the data
    batch_size, height, width, num_ori = data.shape

    # Create output array
    blurred_data = np.zeros_like(data)

    # Process each map separately
    for i in range(batch_size):
        # Pad the spatial dimensions with 1 pixel of zeros
        padded_data = np.pad(
            data[i],
            (
                (1, 1),
                (1, 1),
                (0, 0),
            ),  # Pad height and width with 1, don't pad orientation
            mode="constant",
            constant_values=constant_values,
        )

        # Apply Gaussian filter with appropriate boundary conditions
        padded_blurred = gaussian_filter(
            padded_data,
            sigma=sigma,
            mode=["nearest", "nearest", "wrap"],  # Modes for height, width, orientation
        )

        # Remove the padding to get back to original size
        blurred_data[i] = padded_blurred[1:-1, 1:-1, :]

    return blurred_data
