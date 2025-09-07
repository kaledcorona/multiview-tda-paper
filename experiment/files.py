"""
Module Description:
-------------------
Functions that reads a CSV file with .

Author: Kaled Corona
Date: 2025-02-17
"""

# ============================
# Standard Library Imports
# ============================
from pathlib import Path
from typing import Dict, TypeAlias

import os

from datetime import datetime
from pathlib import Path

from typing import Tuple, List, Dict, Generator, Any
# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd


# ============================
# Local Application Imports
# ============================
#from my_project.utils import helper_function
#from my_project.models import User
#from my_project.config import SETTINGS

# ============================
# Configuration & Settings
# ============================

# Maps model_id to confusion matrix
ConfusionMatrices: TypeAlias = Dict[str, np.ndarray] 

# Maps filename to ConfusionMatrices 
FileConfusionMatrices: TypeAlias = Dict[str, ConfusionMatrices] 

# Classification metrics
ClassificationMetrics: TypeAlias = Dict[str, float]

# Classification metrics per model
ModelMetrics: TypeAlias = Dict[str, ClassificationMetrics] 

# Classification metrics per file
FileMetrics: TypeAlias = Dict[str, ModelMetrics] 

# ============================
# Main Execution 
# ============================

"""
#TODO: Falta modificar
def finalize_and_save_results(results: Dict[str, Dict[str, Any]], 
                              experiment_dir: str, current_time) -> None:
    #logger.info("Saving results (metrics, predictions, etc.)...")
    output_csv = os.path.join(experiment_dir, f"metrics_{current_time}.csv")
    save_detailed_metrics(results, output_csv)
    #logger.info(f"Saved detailed metrics to {output_csv}")

    pred_csv = os.path.join(experiment_dir, f"predictions_{current_time}.csv")
    save_predictions(results, pred_csv)
    #logger.info(f"Saved predictions to {pred_csv}")

"""
"""
#TODO: faltante modificar
def save_predictions(results: Dict[str, Dict[str, Any]], output_csv: str) -> None:
    #
    Save predictions and ground truth from a nested dictionary to a CSV file.

    This function extracts predictions and ground truth for each model combination
    from the results dictionary and flattens them into a CSV file for analysis.


    Args:
        results (Dict[str, Dict[str, Any]]): A nested dictionary with the following structure:
            {
                "model_id_1": {
                    "metrics": [...],
                    "predictions": [...],
                    "ground_truth": [...]
                },
                "model_id_2": {
                    "metrics": [...],
                    "predictions": [...],
                    "ground_truth": [...]
                },
                ...
            }
        output_csv (str): Path to the output CSV file.

    Returns:
        None: Saves the extracted data to the specified CSV file.

    Raises:
        ValueError: If the length of predictions and ground truth do not match for any model.

    Example:
        >>> results = {
        ...     "model_1": {
        ...         "metrics": [],
        ...         "predictions": [1, 0, 1],
        ...         "ground_truth": [1, 1, 0]
        ...     },
        ...     "model_2": {
        ...         "metrics": [],
        ...         "predictions": [0, 0, 1],
        ...         "ground_truth": [0, 0, 1]
        ...     }
        ... }
        >>> save_results_to_csv(results, "output.csv")
    #
    rows = []

    # Iterate through the results dictionary
    for model_id, data in results.items():
        predictions = data.get("predictions", [])
        ground_truth = data.get("ground_truth", [])

        # Ensure predictions and ground truth lengths match
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch for model '{model_id}': "
                f"{len(predictions)} predictions vs {len(ground_truth)} ground truth labels."
            )

        # Flatten the data
        for pred, true in zip(predictions, ground_truth):
            rows.append({"model_id": model_id, "model_prediction": pred, "true_label": true})

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
"""

def ensure_dir_exists(dir: Path = Path('dir')) -> None:
    """Ensure directory exists, if not create it"""
    dir.mkdir(parents=True, exist_ok=True)


def generate_filename(path: Path = Path('./'),
                      name: str = "file",
                      format: str = "csv") -> Path:
    """
    Generates a timestamped filename in the specified directory.

    Args:
        path (Path, optional): The directory where the file will be created. Defaults to the current directory ('./').
        name (str): The base name for the file. Defaults to "file".
        format (str): The file extension (format). Defaults to "csv".

    Returns:
        Path: The full path to the generated file.

    Example:
        >>> generate_filename(path=Path("/tmp"), name="report", format="txt")
        Positional output: Path("/tmp/report_2023-10-14_10-30.txt")
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (path.resolve() / f"{name}_{current_time}.{format}")


def find_matching_files(directory: Path, keyword: str) -> set[Path]:
    """
    Finds all filenames in a directory that start with a given keyword.

    Args:
        directory (Path): The path to the directory.
        keyword (str): The keyword to match at the start of filenames.

    Returns:
        set[Path]: A set of matching file paths.
    """
    return {
        file
        for file in directory.iterdir()
        if file.name.startswith(keyword) and file.suffix == ".csv"
    }


def read_csv_in_chunks(file_path: Path,
                       chunk_size: int,
                       chunk_type: str = 'uint8',
                      ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Reads a CSV file in chunks, yielding features and labels as NumPy arrays.

    This generator reads the file lazily, extracting features from all columns 
    except the last and labels from the last column.

    Args:
        file_path (Path): Path to the CSV file.
        chunk_size (int): Number of rows to read per chunk.
        chunk_type (str): Data type to convert to (e.g., 'uint8', 'float32'). 
            Must be a valid NumPy dtype.

    Yields:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Features (X_chunk): A 2D array of shape (chunk_size, n_features).
            - Labels (y_chunk): A 1D array of shape (chunk_size,).

    Examples:
        Basic usage:
            >>> for X_chunk, y_chunk in read_csv_in_chunks(file_path, CHUNK_SIZE):
                    print(X_chunk)
"""
    if chunk_type not in np.sctypeDict:
        raise ValueError(f"Invalid chunk_type: {chunk_type}")

    yield from (
        (
            chunk.iloc[:, :-1].to_numpy(dtype=chunk_type), 
            chunk.iloc[:, -1].to_numpy(dtype=chunk_type)
        )
        for chunk in pd.read_csv(file_path, chunksize=chunk_size)
    )
