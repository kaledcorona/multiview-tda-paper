"""
Module Description:
-------------------

This module implements a pipeline for topological data analysis (TDA) on images, leveraging persistent homology and filtration methods. It integrates various filtration techniques, including height, radial, and density filtrations, and computes persistence diagrams using `gtda`'s CubicalPersistence and related transformations. The module supports parallelized execution to efficiently handle large datasets and employs feature extraction methods, such as persistence entropy and amplitude metrics, to create a unified feature set suitable for machine learning tasks.

Key components include:
- **Filtration Techniques:** Radial, height, and density filtrations.
- **Persistence Diagrams:** Computed via cubical persistence, binarization, and scaling.
- **Feature Extraction:** Persistence entropy and amplitude-based features for machine learning models.

Author: me
Date: 2025-02-17
"""

# ============================
# Standard Library Imports
# ============================
from typing import List, Tuple, Dict, Any, TypeAlias, Union

# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd

from sklearn.pipeline import make_union, make_pipeline, FeatureUnion

from gtda.images import Binarizer, RadialFiltration, HeightFiltration, DensityFiltration, DilationFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, Amplitude, PersistenceEntropy

# ============================
# Local Application Imports
# ============================

# ============================
# Configuration & Settings
# ============================

MetricParams = Dict[str, Union[int, float]]
MetricEntry = Dict[str, Union[str, MetricParams]]

# ============================
# Main Execution 
# ============================

# Break up build_tda_pipeline into smaller, focused functions.
def build_tda_pipeline(num_jobs: int = 3) -> FeatureUnion:
    """
    Build a TDA pipeline for computing topological features on images.

    This function assembles all steps for the TDA pipeline, including filtration, persistent homology, and feature extraction, with parallelization support.

    Args:
        n_jobs (int): Number of parallel jobs to run.

    Returns:
          sklearn.pipeline.FeatureUnion: A combined pipeline for further processing.
    """
    radial_centers = get_radial_filtration_coordinates()
    height_vector = get_height_filtration_directions()
 
    filtrations = assemble_filtration_steps(
        height_vector, radial_centers, num_jobs
    )

    persistent_diagrams = assemble_persistence_diagrams_steps(
        filtrations, num_jobs=num_jobs
    )

    features = create_union_features(get_amplitude_metrics(), num_jobs)

    return make_union(
        *[make_pipeline(*diagram_step, features) for diagram_step in persistent_diagrams], n_jobs=num_jobs
    )


def create_union_features(metrics: List[Dict[str, Any]],
                          num_jobs: int,
                          verbose=False):
    """
    Create a union of feature transformations from entropy and amplitude features.

    This function generates a combined feature set by applying two distinct transformations:
    - Persistence Entropy: An entropy-based feature computed using the `PersistenceEntropy` class.
    - Amplitude: A set of amplitude-based features computed using the `Amplitude` class, one for each metric provided.

    These features are then combined using `make_union`, which creates a unified feature transformation pipeline 
    for further use in a machine learning workflow.

    Args:
        metrics (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents the parameters for
                                        creating an `Amplitude` feature. Each dictionary should contain the necessary
                                        keys and values for the instantiation of the `Amplitude` class.
        num_jobs (int): The number of parallel jobs to use when creating features.
        verbose (bool, optional): Whether to print detailed logs during the feature creation process. Defaults to False.

    Returns:
        sklearn.pipeline.FeatureUnion: A `FeatureUnion` object combining the persistence entropy and amplitude features,
                                       ready for use in a machine learning pipeline.
    """
    entropy_features = PersistenceEntropy(n_jobs=num_jobs)
    amplitude_features = [Amplitude(**metric, n_jobs=num_jobs) for metric in metrics]
    
    all_features = [entropy_features] + amplitude_features 
    
    return make_union(*all_features, n_jobs=num_jobs, verbose=verbose)


def get_radial_filtration_coordinates() -> np.ndarray:
    """
    Generate a unique set of grid coordinates for radial filtration.

    This function returns a 9x2 NumPy array, where each row represents 
    a unique (x, y) coordinate pair forming a structured 3x3 grid.

    Reference:
        - Garin and Tauzin (2019), “A Topological ‘Reading’ Lesson.”

    Returns:
        np.ndarray: A (9, 2) array of unique (x, y) coordinates:
            [[ 6,  6], [13,  6], [20,  6],
             [ 6, 13], [13, 13], [20, 13],
             [ 6, 20], [13, 20], [20, 20]]
    """
    grid_values = (6, 13, 20)
    return np.array(
        [[x, y] for x in grid_values for y in grid_values], dtype=np.uint8
    )
    

def get_height_filtration_directions() -> np.ndarray:
    """
    Generate a unique set of vectors for height filtration.

    This function returns a NumPy array of vectors representing movement 
    in various directions (horizontal, vertical, and diagonal) on a grid. 
    The vectors exclude the (0, 0) vector, which represents no movement, 
    and each vector has a unit length of 1.

    Reference:
        - Garin and Tauzin (2019), “A Topological ‘Reading’ Lesson.”

    Returns:
        np.ndarray: A (8, 2) array of 8 unique direction vectors. Each vector 
        is a 2D coordinate with values ranging from -1 to 1, excluding (0, 0):
        [
            [-1, -1], [-1,  0], [-1,  1],
            [ 0, -1], [ 0,  1],
            [ 1, -1], [ 1,  0], [ 1,  1]
        ]

    Notes:
    - All direction vectors have a norm of 1.
    - Uses np.int8 for memory efficiency.
    """
    directions = [
        [dx, dy]
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1] 
        if (dx, dy) != (0, 0)
    ]
    return np.array(directions, dtype=np.int8)


def assemble_filtration_steps(height_vector: np.ndarray,
                            radial_centers: np.ndarray,
                            num_jobs: int,
                            density_radii: List[int] = [2,4,6]
                           ) -> List[Any]:
    """
    Generates filtration tasks to extract topological features.

    This function creates tasks for various types of filtration. 
    Each task type is instantiated for the corresponding input elements.

    Args:
        height_vector (np.ndarray): A 2D array for height filtration.
        radial_centers (np.ndarray): A 2D array for radial filtration.
        density_radii (List[int]): A list of radii for density filtration.
        num_jobs (int): The number of parallel jobs.

    Returns:
        List[Any]: A list of filtration task instances:
            - One `HeightFiltration` task per vector in `height_vector`.
            - One `RadialFiltration` task per center in `radial_centers`.
            - One `DensityFiltration` task per radius in `density_radii`.
            - A single `DilationFiltration` task.
    """
    height_filtration_task = [
        HeightFiltration(vector, num_jobs) 
        for vector in height_vector
    ]

    radial_filtration_task = [
        RadialFiltration(center, n_jobs=num_jobs) 
        for center in radial_centers
    ]

    density_filtration_task = [
        DensityFiltration(steps, n_jobs=num_jobs)
        for steps in density_radii]

    dilation_filtration_task = [DilationFiltration(n_jobs=num_jobs)]

    all_filtration_tasks = (
        height_filtration_task +
        radial_filtration_task +
        density_filtration_task +
        dilation_filtration_task 
    )
    
    return all_filtration_tasks


def assemble_persistence_diagrams_steps(filtrations: List[Any],
                                        num_jobs: int,
                                        binary_threshold: float = 0.4) -> List[Any]:
    """
    Generates a list of processing steps to compute persistence diagrams for each filtration.

    For each filtration, the steps include:
    - Binarization (with a configurable threshold),
    - Filtration,
    - Cubical persistence computation,
    - Scaling.

    Args:
        filtrations (List[Any]): A list of filtrations to process.
        binary_threshold (float): Threshold for binarization. Default is 0.4.
        num_jobs (int): Number of parallel jobs for computation.

    Returns:
        List[Any]: A list of processing steps for each filtration, where each step is
                   a sequence of [Binarizer, Filtration, CubicalPersistence, Scaler].
    """
    def workflow_persistent_diagram(filtration: Any) -> List[Any]:
        binarizer = Binarizer(binary_threshold, n_jobs=num_jobs)
        persistence = CubicalPersistence(n_jobs=num_jobs)
        scaler = Scaler(n_jobs=num_jobs)
        return [binarizer, filtration, persistence, scaler]

    return [
        workflow_persistent_diagram(filtration)
        for filtration in filtrations
    ]


def get_amplitude_metrics() -> List[MetricEntry]:
    """Returns a list of amplitude metrics for vectorization."""
    metrics : List[Tuple[str, Dict[str, Union[int, float]]]] = [
        ("bottleneck", {}),
        ("wasserstein", {"p": 1}),
        ("wasserstein", {"p": 2}),
        ("landscape", {"p": 2}),
        ("betti", {"p": 2}),
        ("heat", {"sigma": 1.5}),
    ]
    return [{"metric": metric, "metric_params": params} for metric, params in metrics]
