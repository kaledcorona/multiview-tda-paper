"""
Module Description:
-------------------
This module contains functions that perform various image processing and feature extraction tasks. 
It includes utilities for reshaping image data into square matrices, splitting images into quadrants, 
and generating combinations of feature views for further processing. The module utilizes NumPy for efficient 
array manipulation and supports flexible configurations for reshaping and padding image data.

Functions include:
- `reshape_to_square_image`: Reshapes a 2D NumPy array into a square image shape.
- `split_and_reshape_quadrants`: Splits images into quadrants and reshapes them into 1D arrays.
- `select_and_combine_views`: Selects and combines feature matrices based on specified views.
- `generate_view_combinations`: Generates combinations of views that include all required views.

Author: me
Date: 2025-02-21
"""


# ============================
# Standard Library Imports
# ============================
import itertools
from math import isqrt, ceil, sqrt
from typing import Dict, TypeAlias, Iterator, List
from pathlib import Path

# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Local Application Imports
# ============================
#from my_project.utils import helper_function
#from my_project.models import User
#from my_project.config import SETTINGS
from experiment.files import generate_filename

# ============================
# Configuration & Settings
# ============================

# A view in multiview
View: TypeAlias = str

# ============================
# Main Execution 
# ============================


def validate_input(array: np.ndarray) -> None:
    """Ensure the input array has the correct dimensions."""
    if array.ndim != 2:
        raise ValueError(
            f"Input must have shape (n_samples, n_features), but instead: {array.shape}."
        )
        

def calculate_side_length(n_features: int) -> int:
    """Calculate the side length for reshaping."""
    return (
        isqrt(n_features) 
        if isqrt(n_features) ** 2 == n_features 
        else ceil(sqrt(n_features))
    )


def pad_array_if_needed(array: np.ndarray, side_length: int, pad_zeros: bool) -> np.ndarray:
    """Pad the array with zeros if the number of features is not a perfect square."""
    n_samples, n_features = array.shape
    if side_length ** 2 == n_features:
        return array
    
    if not pad_zeros:
        raise ValueError(f"Number of features ({n_features}) is not a perfect square, and padding is not enabled.")
    
    padded_array = np.pad(
        array,
        ((0, 0), (0, side_length ** 2 - n_features)),
        mode="constant"
    )
    return padded_array


def reshape_to_square_image(array: np.ndarray, pad_zeros: bool = False) -> np.ndarray:
    """
    Reshape a 2D NumPy array into (n_samples, sqrt(n_features), sqrt(n_features)).

    Args:
        array (np.ndarray): Input NumPy array with shape (n_samples, n_features).
        pad_zeros (bool, optional): If True, pads zeros to make n_features a perfect square.

    Returns:
        np.ndarray: Reshaped array with shape (n_samples, sqrt(n_features), sqrt(n_features)).

    Raises:
        ValueError: If n_features is not a perfect square and pad_zeros is False.

    Examples:
        >>> X_flat = np.random.rand(100, 256)  # 256 = 16^2
        >>> X_reshaped = reshape_to_square_image(X_flat)
        >>> print(X_reshaped.shape)
        (100, 16, 16)
    """   
    validate_input(array)
    n_samples, n_features = array.shape
    side_length = calculate_side_length(n_features)
    
    padded_array = pad_array_if_needed(array, side_length, pad_zeros)
    
    return padded_array.reshape(n_samples, side_length, side_length)


def split_and_reshape_quadrants(images: np.ndarray) -> dict[View, np.ndarray]:
    """
    Splits each image into four quadrants and reshapes them into 1D arrays.

    This function first splits each image into four quadrants and then reshapes 
    each quadrant from (n_samples, height/2, width/2) to (n_samples, 196), where 
    196 is the flattened size of each quadrant.

    Args:
        images (np.ndarray): Array of images with shape (n_samples, height, width).

    Returns:
        dict[str, np.ndarray]: A dictionary containing the four quadrants reshaped to 1D.
    """
    quadrants = split_to_quadrants(images)
    return reshape_quadrants(quadrants, len(images))


def reshape_quadrants(quadrants: dict[View, np.ndarray],
                      num_samples: int) -> dict[View, np.ndarray]:
    """
    Reshapes the quadrants from (n_samples, height/2, width/2) to (n_samples, 196).

    This function takes a dictionary of quadrants and reshapes each one from its 
    original shape of (n_samples, height/2, width/2) to a flattened shape of 
    (n_samples, 196), where 196 is the size of each 14x14 quadrant.

    Args:
        quadrants (dict[str, np.ndarray]): A dictionary of quadrants with shape (n_samples, height/2, width/2).
        num_samples (int): The number of samples (images) in the dataset.

    Returns:
        dict[str, np.ndarray]: A dictionary of reshaped quadrants with shape (n_samples, 196).
    """
    return {key: quadrant.reshape(num_samples, -1) for key, quadrant in quadrants.items()}


def split_to_quadrants_pure(images: np.ndarray) -> dict[View, np.ndarray]:
    """
    Splits each image into four quadrants.

    This function takes an array of images and divides each image into four quadrants:
    top-left, top-right, bottom-left, and bottom-right. It assumes that the images 
    have even dimensions, and it splits the images equally along both axes.

    Args:
        images (np.ndarray): A NumPy array of images with shape (n_samples, height, width).
                              The height and width must be even.

    Returns:
        Dict[View, np.ndarray]: A dictionary with four entries:
            - 'top_left': Top-left quadrant of each image.
            - 'top_right': Top-right quadrant of each image.
            - 'bottom_left': Bottom-left quadrant of each image.
            - 'bottom_right': Bottom-right quadrant of each image.

    Example:
        >>> images = np.random.rand(100, 32, 32)  # 100 images of size 32x32
        >>> quadrants = split_to_quadrants_pure(images)
        >>> quadrants["top_left"].shape
        (100, 16, 16)
        >>> quadrants["bottom_right"].shape
        (100, 16, 16)

    Note:
        This function assumes that the images are already validated.
    """
    height, width = images.shape[1], images.shape[2]
    
    half_height = height // 2
    half_width = width // 2
    
    return {
        "top_left": images[:, :half_height, :half_width],
        "top_right": images[:, :half_height, half_width:],
        "bottom_left": images[:, half_height:, :half_width],
        "bottom_right": images[:, half_height:, half_width:]
    }


def split_to_quadrants(images: np.ndarray) -> dict[View, np.ndarray]:
    """
    Validates and splits images into four equal quadrants.

    This function ensures that the input images have even dimensions before 
    delegating the actual splitting to `split_to_quadrants_pure`. Each image 
    is divided into four quadrants: top-left, top-right, bottom-left, and bottom-right.

    Args:
        images (np.ndarray): A NumPy array of images with shape (n_samples, height, width).
                             The height and width must be even.

    Returns:
        Dict[View, np.ndarray]: A dictionary mapping quadrant names to their respective 
                                NumPy arrays.

    Raises:
        ValueError: If the input images do not have even height and width.

    Example:
        >>> images = np.random.rand(100, 32, 32)  # 100 images of size 32x32
        >>> quadrants = split_to_quadrants(images)
        >>> quadrants["top_left"].shape
        (100, 16, 16)

    Note:
        This function does not perform the actual image splitting but ensures that 
        dimensions are valid before calling `split_to_quadrants_pure`.
    """
    height, width = images.shape[1], images.shape[2]
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Image dimensions ({height}, {width}) must be even.")
    return split_to_quadrants_pure(images)
    
    
def select_and_combine_views(original_features: np.ndarray,
                             tda_features: np.ndarray,
                             quadrant_features: dict[View, np.ndarray],
                             selected_views: set[View]) -> dict[View, np.ndarray]:
    """
    Selects and combines feature matrices based on specified views.

    This function returns a dictionary of feature matrices corresponding to the
    selected views.

    Args:
        original_features (np.ndarray): Feature matrix for the original images.
        tda_features (np.ndarray): Feature matrix for TDA features.
        quadrant_features (Dict[str, np.ndarray]): Dictionary of quadrant feature matrices.
        selected_views (set[View]): A set of selected views to include in the result.

    Returns:
        Dict[str, np.ndarray]: A dictionary where each key is a selected view and the value is
                                the corresponding feature matrix.

    Example:
        >>> result = select_and_combine_views(original_features, tda_features, quadrant_features, selected_views)

    Note:
        Time complexity: O(n), where n is the number of selected views.
    """
    view_mapping = {
        "original": original_features,
        "tda": tda_features,
        "top_left": quadrant_features.get("top_left", np.empty((0, 0))),
        "top_right": quadrant_features.get("top_right", np.empty((0, 0))),
        "bottom_left": quadrant_features.get("bottom_left", np.empty((0, 0))),
        "bottom_right": quadrant_features.get("bottom_right", np.empty((0, 0))),
    }
    return {view: view_mapping[view] for view in selected_views}


def generate_view_combinations(views: set[View], required_views: set[View]) -> Iterator[set[View]]:
    """
    Generates non-empty combinations of views that include all the required_views.
    It yields each combination, ensuring efficient memory usage for large datasets.

    Args:
        views (set[str]): A set of view names to combine.
        required_views (set[str]): A set of view names that must appear in each combination. 

    Returns:
        Iterator[set[str]]: An iterator that yields sets of views.

    Examples:
        Basic usage example:
            >>> views = {"original", "tda", "top_left"}
            >>> required_views = {"original"}
            >>> result = generate_view_combinations(views, required_views)
            >>> print(list(result))
            [{'original'}, {'original', 'top_left'}, {'original', 'tda', 'top_left'}]

    Note:
        Time complexity: O(n^2) where n is len(views - required_views)
    """
    remaining_views = views - required_views
    
    for cardinality in range(0, len(remaining_views) + 1):
        for combo in itertools.combinations(remaining_views, cardinality):
            yield required_views.union(combo)
        
    

def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Saves a single grayscale image to the given path.
    
    Ensures the image is 2D before saving.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Fix shape if needed ---
    if image.ndim == 1:
        side = int(np.sqrt(image.shape[0]))
        if side * side != image.shape[0]:
            raise ValueError(f"Cannot reshape array of size {image.shape[0]} into square image")
        image = image.reshape((side, side))
    
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D (grayscale), but got shape {image.shape}")
    
    plt.imsave(output_path, image, cmap="gray")




def save_sample_views(
    views: dict[str, np.ndarray],
    base_output_dir: Path,
    experiment_name: str,
    num_samples: int = 20
) -> None:
    """
    Saves a few sample images from each view into organized folders.

    Args:
        views (dict[str, np.ndarray]): Mapping of view names to image batches.
        base_output_dir (Path): Base folder where images are saved.
        num_samples (int): Number of samples to save per view.
    """
    for view_name, images in views.items():
        view_dir = base_output_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)

        # Save limited number of images
        for idx, image in enumerate(images[:num_samples]):
            
            filename = generate_filename(
            path=view_dir,
            name=experiment_name,
            format="png")
            
            output_path = view_dir / f"{filename.stem}_{idx}.png"
            save_image(image, output_path)

