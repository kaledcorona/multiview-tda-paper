"""
Module Description:
-------------------

Description

Key components include:

Author: me
Date: 2025-0x-xx
"""

# ============================
# Standard Library Imports
# ============================
from typing import Optional, Tuple, Callable


from concurrent.futures import ProcessPoolExecutor
# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd

# ============================
# Local Application Imports
# ============================
from experiment.view import (
    reshape_to_square_image, 
)
# ============================
# Configuration & Settings
# ============================

# ============================
# Main Execution 
# ============================




def add_lines_noise_to_images(
    image: np.ndarray, 
    number_lines: int,
    random_seed: int,
    max_line_height: int = 100,
    min_line_height: int = 3,
    line_transparency: float = 0.7,
) -> np.ndarray:
    """
    Add random line noise to a single grayscale image (2D numpy array).
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale image to be modified.
    number_lines : int
        Number of lines to add to the image.
    random_seed : int
        Seed for reproducibility of random line generation.
    max_line_height : int
        Maximum length of a line (in pixels).
    min_line_height : int
        Minimum length of a line (in pixels).
    line_transparency : float, optional
        Fraction of the image's maximum intensity for the line brightness 
        (e.g., 1.0 means use the max pixel value in the image). Defaults to 1.0.
    
    Returns
    -------
    np.ndarray
        A copy of the original image with random lines applied.
    
    Notes
    -----
    - Lines are drawn in random orientations. 
    - If parts of a line fall outside the image boundaries, those parts are skipped.
    - The parameter 'line_transparency' scales the line's pixel intensity 
      relative to the image's maximum value.
    """
    if image.ndim != 2:
        raise ValueError("Only 2D grayscales images are supported.")
    if not (0.0 <= line_transparency <= 1.0):
        raise ValueError("line_transparency must be between 0 and 1.")
    if min_line_height > max_line_height:
        raise ValueError("min_line_height must be <= max_line_height.")



    
    # Fix the random seed for reproducibility
    np.random.seed(random_seed)

    # Copy image to avoid modifying in-place
    image_copy = image.copy()
    height, width = image_copy.shape

    # Determine the brightness for the lines
    max_value = image_copy.max()
    line_brightness = max_value * line_transparency

    # Generate the specified number of lines
    for _ in range(number_lines):
        # Random starting point for the line
        start_row = np.random.randint(0, height)
        start_col = np.random.randint(0, width)

        # Random line length
        line_length = np.random.randint(min_line_height, max_line_height + 1)
        # Random angle in [0, 2π)
        angle = np.random.uniform(0, 2 * np.pi)

        # Draw the line pixel by pixel along the chosen angle
        for step in range(line_length):
            # Compute the next point in floating coordinates
            row_offset = step * np.sin(angle)
            col_offset = step * np.cos(angle)

            # Round to nearest pixel
            current_row = int(round(start_row + row_offset))
            current_col = int(round(start_col + col_offset))

            # Check boundaries
            if (0 <= current_row < height) and (0 <= current_col < width):
                image_copy[current_row, current_col] = line_brightness
            else:
                # If we're out of bounds, stop drawing this line
                break

    return image_copy

def _add_lines_noise_unpack(args: Tuple[np.ndarray, int, int, int, int, float]) -> np.ndarray:
    return add_lines_noise_to_images(*args)


def add_lines_noise_to_batch(
    images: np.ndarray,
    number_lines: int,
    random_seed: int,
    max_line_height: int = 100,
    min_line_height: int = 3,
    line_transparency: float = 0.7,
    max_workers: Optional[int] = 5,
) -> np.ndarray:
    """
    Apply random line noise to a batch of grayscale images (3D numpy array).

    Parameters
    ----------
    images : np.ndarray
        A 3D numpy array of shape (N, H, W) representing grayscale images.
    number_lines : int
        Number of lines to draw per image.
    random_seed : int
        Base seed for reproducibility; each image gets a unique derived seed.
    max_line_height : int
        Maximum line length in pixels.
    min_line_height : int
        Minimum line length in pixels.
    line_transparency : float
        Fraction of max pixel intensity for line brightness.
    max_workers : Optional[int]
        Maximum number of parallel workers. Defaults to CPU count.

    Returns
    -------
    np.ndarray
        A new array with the same shape as `images`, with line noise added.
    """
    if images.ndim != 3:
        raise ValueError(f"Expected input shape (N, H, W) for batch of grayscale images. Dimension {images.ndim}")

    n_images = images.shape[0]
    base_seed = int(random_seed)
    seeds = [base_seed + (i * 2) for i in range(n_images)]

    # Prepare argument tuples for each image
    args = [
        (images[i], number_lines, seeds[i], max_line_height, min_line_height, line_transparency)
        for i in range(n_images)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        noisy_images = list(executor.map(_add_lines_noise_unpack, args))

    return np.stack(noisy_images)


def add_circles_noise_to_images(
    image: np.ndarray, 
    number_circles: int,
    random_seed: int,
    max_radius: int = 10,
    min_radius: int = 2,
    circle_transparency: float = 0.7,
) -> np.ndarray:
    """
    Add random circle noise to a single grayscale image (2D numpy array).
    
    Parameters
    ----------
    image : np.ndarray
        2D grayscale image to be modified.
    number_circles : int
        Number of circles to add to the image.
    random_seed : int
        Seed for reproducibility of random circle generation.
    max_radius : int
        Maximum radius of a circle (in pixels).
    min_radius : int
        Minimum radius of a circle (in pixels).
    circle_transparency : float, optional
        Fraction of the image's maximum intensity for the circle brightness 
        (e.g., 1.0 means use the max pixel value in the image). Defaults to 1.0.
    
    Returns
    -------
    np.ndarray
        A copy of the original image with random circles applied.
    
    Notes
    -----
    - Circles are filled and drawn at random positions.
    - Circles outside the image boundaries are clipped.
    - The parameter 'circle_transparency' scales the pixel intensity 
      relative to the image's maximum value.
    """
    if image.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")
    if not (0.0 <= circle_transparency <= 1.0):
        raise ValueError("circle_transparency must be between 0 and 1.")
    if min_radius > max_radius:
        raise ValueError("min_radius must be <= max_radius.")

    np.random.seed(random_seed)

    image_copy = image.copy()
    height, width = image_copy.shape
    max_value = image_copy.max()
    circle_brightness = max_value * circle_transparency

    for _ in range(number_circles):
        center_y = np.random.randint(0, height)
        center_x = np.random.randint(0, width)
        radius = np.random.randint(min_radius, max_radius + 1)

        # Draw a filled circle (naive raster approach)
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                if 0 <= x < width and 0 <= y < height:
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                        image_copy[y, x] = circle_brightness

    return image_copy

def _add_circles_noise_unpack(args: Tuple[np.ndarray, int, int, int, int, float]) -> np.ndarray:
    return add_circles_noise_to_images(*args)

def add_circles_noise_to_batch(
    images: np.ndarray,
    number_circles: int,
    random_seed: int,
    max_radius: int = 5,
    min_radius: int = 2,
    circle_transparency: float = 0.7,
    max_workers: Optional[int] = 5,
) -> np.ndarray:
    """
    Apply random circle noise to a batch of grayscale images (3D numpy array).

    Parameters
    ----------
    images : np.ndarray
        A 3D numpy array of shape (N, H, W) representing grayscale images.
    number_circles : int
        Number of circles to draw per image.
    random_seed : int
        Base seed for reproducibility; each image gets a unique derived seed.
    max_radius : int
        Maximum radius in pixels.
    min_radius : int
        Minimum radius in pixels.
    circle_transparency : float
        Fraction of max pixel intensity for circle brightness.
    max_workers : Optional[int]
        Maximum number of parallel workers. Defaults to 5.

    Returns
    -------
    np.ndarray
        A new array with the same shape as `images`, with circle noise added.
    """
    if images.ndim != 3:
        raise ValueError(f"Expected input shape (N, H, W) for batch of grayscale images. Got {images.ndim}D.")

    n_images = images.shape[0]
    seeds = [random_seed + (i * 2) for i in range(n_images)]

    args = [
        (images[i], number_circles, seeds[i], max_radius, min_radius, circle_transparency)
        for i in range(n_images)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        noisy_images = list(executor.map(_add_circles_noise_unpack, args))

    return np.stack(noisy_images)


import numpy as np

def add_salt_and_pepper_noise(
    image: np.ndarray,
    p: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Apply salt-and-pepper noise to a single 2-D grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D array representing the grayscale image.
    p : float
        Overall corruption probability (0 ≤ p ≤ 1).  
        Each pixel becomes:
            0            with probability p / 2   (pepper)
            max(image)   with probability p / 2   (salt)
            unchanged    with probability 1 – p
    random_seed : int | None, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy copy of the original image.
    """
    # --- validation --------------------------------------------------------
    if image.ndim != 2:
        raise ValueError("Only 2-D grayscale images are supported.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0, 1].")

    # --- RNG ---------------------------------------------------------------
    rng = np.random.default_rng(random_seed)

    # --- prepare -----------------------------------------------------------
    image_noisy = image.copy()
    max_val     = image.max()

    # --- draw a uniform [0,1] mask once ------------------------------------
    mask = rng.random(size=image.shape)

    # pepper:   mask < p/2
    # salt:     mask > 1 - p/2
    # untouched elsewhere
    image_noisy[mask <  p / 2]      = 0
    image_noisy[mask > 1 - p / 2]   = max_val

    return image_noisy


def _add_salt_pepper_noise_unpack(
    args: Tuple[np.ndarray, float, int]
) -> np.ndarray:
    """
    Convenience shim for executor.map.
    """
    return add_salt_and_pepper_noise(*args)

def add_salt_and_pepper_noise_batch(
    images: np.ndarray,
    p: float,
    random_seed: int,
    max_workers: Optional[int] = 5,
) -> np.ndarray:
    """
    Vector of shape (N, H, W)  →  noisy copy, same shape.

    Parameters
    ----------
    images : np.ndarray
        Batch of grayscale images.
    p : float
        Corruption probability (see single-image function).
    random_seed : int
        Master seed; each image gets a unique offset.
    max_workers : Optional[int]
        Parallel workers for `ProcessPoolExecutor`.

    Returns
    -------
    np.ndarray
        Noisy batch.
    """
    if images.ndim != 3:
        raise ValueError("Expected array of shape (N, H, W).")

    n_images = images.shape[0]
    seeds    = [random_seed + 2 * k for k in range(n_images)]

    args = [(images[k], p, seeds[k]) for k in range(n_images)]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        noisy = list(ex.map(_add_salt_pepper_noise_unpack, args))

    return np.stack(noisy, axis=0)


def add_salt_pepper_noise_to_batch_wrapper(
    images: np.ndarray,
    p: float,                 # plays the role of “quantity”
    random_seed: int,
) -> np.ndarray:
    """
    Thin adapter: keeps the original (images, quantity, random_state)
    calling convention while delegating to the new implementation.
    """
    return add_salt_and_pepper_noise_batch(images, p, random_seed)


def contaminate_dataset(
    data: np.ndarray,
    noise_fn: Callable[[np.ndarray, int, float], np.ndarray],
    quantity: int,
    transparency: float,
    random_state: int
    ) -> np.ndarray:
    """
    Applies a noise function to the dataset after reshaping it to square images.
    
    Args:
    data (np.ndarray): The dataset as 1D flattened images.
    noise_fn (Callable): The noise function to apply.
    quantity (int): Amount of noise elements (lines, circles, points).
    transparency (float): Transparency of the added noise.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    np.ndarray: The contaminated dataset, reshaped back to 1D arrays.
    """
    n_samples = data.shape[0]
    images = reshape_to_square_image(data)
    noisy_images = noise_fn(images, quantity, random_state)
    return noisy_images.reshape(n_samples, -1)


def add_noise_to_data(
    X_chunk: np.ndarray,
    noise_type: str,
    quantity: float,         
    random_state: int,
):
    noise_functions: dict[str, Callable] = {
        "lines"       : add_lines_noise_to_batch,
        "circles"     : add_circles_noise_to_batch,
        "salt_pepper" : add_salt_pepper_noise_to_batch_wrapper, # new entry
    }

    if noise_type not in noise_functions:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return contaminate_dataset(
        X_chunk,
        noise_functions[noise_type],
        quantity,               # either #points or probability p
        transparency=1.0,       # still required by the signature
        random_state=random_state,
    )
