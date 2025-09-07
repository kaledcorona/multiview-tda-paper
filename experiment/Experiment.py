# ============================
# Standard Library Imports
# ============================
from typing import TypeAlias, Iterator, Any, Callable
from pathlib import Path
import logging
import json
import random
import time
import gc

# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ============================
# Local Application Imports
# ============================
from experiment.ExperimentConfig import ExperimentConfig
from experiment.logs import start_experiment_logging, log_experiment_metadata
from experiment.files import ensure_dir_exists, read_csv_in_chunks, generate_filename
from experiment.view import (
    split_and_reshape_quadrants, 
    reshape_to_square_image, 
    select_and_combine_views,
    generate_view_combinations,
    save_sample_views,
)

from experiment.noise import add_noise_to_data
from experiment.model import train_multiview_stacking
from experiment.topological import build_tda_pipeline

# ============================
# Configuration & Settings
# ============================

View: TypeAlias = str


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
        self.logger: logging.Logger = start_experiment_logging(
            self.config.log_dir
        )
        
        self.results: list = []
    
        self.views_combinations: Iterator[set[View]] = (
            generate_view_combinations(
                self.config.views, {"original"}
            )
        )
        self.logger.info("Views combinations successfully generated.")
    
        self.tda_pipeline: FeatureUnion = build_tda_pipeline(
            num_jobs=self.config.num_jobs
        )
        self.logger.info(
            "Topological Data Analysis pipeline generated."
        )
    
        ensure_dir_exists(self.config.results_dir)
        self.logger.info("Folder to save results has been created.")


##########################################
# 3. Model Training
##########################################

    def train_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        train_quadrant_views: dict[View, np.ndarray],
        test_quadrant_views: dict[View, np.ndarray],
        train_tda_features: np.ndarray,
        test_tda_features: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Train models and store predictions & true labels."""
        start_time = time.perf_counter()
    
        for current_view in tqdm(
            self.views_combinations, desc="Models:"
        ):
            model_name = "-".join(current_view)
            self.logger.info(f"Training model: {model_name}")
    
            model_start_time = time.perf_counter()
    
            dataset_train = select_and_combine_views(
                X_train,
                train_tda_features,
                train_quadrant_views,
                current_view
            )
            dataset_test = select_and_combine_views(
                X_test,
                test_tda_features,
                test_quadrant_views,
                current_view
            )
    
            y_pred, y_true = train_multiview_stacking(
                dataset_train,
                dataset_test,
                y_train,
                y_test,
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.num_jobs
            )
    
            model_elapsed_time = time.perf_counter() - model_start_time
            self.logger.info(
                f"Model {model_name} completed in "
                f"{model_elapsed_time:.2f} seconds."
            )
    
            accuracy = accuracy_score(y_true, y_pred)
            self.logger.info(
                f">>> Model {model_name} trained. "
                f"Accuracy: {accuracy:.4f} "
                f"({model_elapsed_time:.2f} sec.)"
            )
    
            self.results.append({
                "model_name": model_name,
                "predictions": y_pred.tolist(),
                "ground_truth": y_true.tolist(),
                "accuracy": accuracy,
                "random_state": self.config.random_state,
                "train_size": self.config.train_split,
                "test_size": self.config.test_split
            })
    
        self.logger.info("All models trained successfully.")
        total_elapsed_time = (time.perf_counter() - start_time) / 60
        self.logger.info(
            f"========== [Model Training Completed] "
            f"in {total_elapsed_time:.2f} minutes =========="
        )

    def save_results_to_json(self, output_file: str) -> None:
        """Save results to a JSON file."""
        output_data = {
            "train_split_percentage": self.config.train_split,
            "noise_enable": self.config.noise_enabled,
            "noise_type": self.config.noise_type,
            "noise_quantity": self.config.noise_quantity,
            "noise_transparency": self.config.noise_transparency,
            "results": self.results
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
    
        self.logger.info(f"Results saved to {output_file}")


##########################################
# 2. Processing
##########################################

    def prepare_multiview_data_chunk(
        self,
        X_chunk: np.ndarray,
        y_chunk: np.ndarray,
    ) -> None:
        """
        Preprocesses data to generate the synthetic views.
        """
        self.logger.info(
            f"Starting multiview data preparation for "
            f"a chunk of size {X_chunk.shape}..."
        )
        self.logger.info(f"Running using {self.config.num_jobs} jobs.")
    
        start_time = time.perf_counter()

        # ==========================================
        # 2.0 Subsample the data set
        # ==========================================
        if self.config.subsample_dataset:
            X_chunk, _, y_chunk, _ = train_test_split(
                X_chunk, 
                y_chunk, 
                train_size=100, 
                random_state=self.config.random_state, 
                stratify=y_chunk
            )
    
        # ==========================================
        # 2.1 Noise Contamination
        # ==========================================
        if not self.config.test_noise:
            X_chunk = self._maybe_add_noise(X_chunk)
    
        # ==========================================
        # 2.2 Data Split
        # ==========================================
        X_train, X_test, y_train, y_test = self._split_data(
            X_chunk, y_chunk
        )

        # ==========================================
        # 2.2.1 Noise Contamination For Testing Set
        # ==========================================
        # This is to test robustness against noise in testing
        if self.config.test_noise:
            X_test = self._maybe_add_noise(X_test)

        
        # ==========================================
        # 2.3 Reshape Data
        # ==========================================
        train_images, test_images = self._reshape_images(
            X_train, X_test
        )
    
        # ==========================================
        # 2.4 Views Generation
        # ==========================================
        self.logger.info("Splitting images into quadrant views...")
    
        train_quadrant_views: dict[View, np.ndarray] = (
            split_and_reshape_quadrants(train_images)
        )
        test_quadrant_views: dict[View, np.ndarray] = (
            split_and_reshape_quadrants(test_images)
        )
    
        self.logger.info(
            f"Quadrant view splitting complete. "
            f"Train views: {len(train_quadrant_views)}, "
            f"Test views: {len(test_quadrant_views)}"
        )

        views_output_dir = Path(self.config.results_dir) / "views"

        #self.logger.info("Saving sample images for each view...")
        #save_sample_views(train_quadrant_views, views_output_dir, self.config.exp_name)
        #self.logger.info("Sample views saved successfully.")
    
        # ==========================================
        # 2.5 TDA Transformations
        # ==========================================
        self.logger.info(
            "Starting TDA pipeline transformation on train data..."
        )
        start_time_tda_train = time.perf_counter()
    
        train_tda_features: np.ndarray = self.tda_pipeline.fit_transform(
            train_images
        )
    
        total_time_tda_train = (
            time.perf_counter() - start_time_tda_train
        ) / 60
        self.logger.info(
            f"TDA transformation for train completed "
            f"({total_time_tda_train:.2f} minutes)."
        )
    
        self.logger.info(
            "Applying TDA pipeline transformation on test data..."
        )
        start_time_tda_test = time.perf_counter()
    
        test_tda_features: np.ndarray = self.tda_pipeline.transform(
            test_images
        )
    
        total_time_tda_test = (
            time.perf_counter() - start_time_tda_test
        ) / 60
        self.logger.info(
            f"TDA transformation for test completed "
            f"({total_time_tda_test:.2f} minutes)."
        )
    
        # ==========================================
        # 2.6 Model Training
        # ==========================================
        self.logger.info(
            "========== [Model Training Start] =========="
        )
        self.train_models(
            X_train, X_test,
            train_quadrant_views, test_quadrant_views,
            train_tda_features, test_tda_features,
            y_train, y_test,
        )
        self.logger.info(
            "========== [Experiment Training Complete] =========="
        )



    def _maybe_add_noise(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Optionally contaminates the data chunk with noise, based on configuration.
    
        Args:
            X_chunk (np.ndarray): The input data chunk.
    
        Returns:
            np.ndarray: The (possibly) contaminated data chunk.
        """
        if not self.config.noise_enabled:
            return X_chunk
    
        self.logger.info(
            f"Contaminating dataset with {self.config.noise_type}, "
            f"quantity={self.config.noise_quantity}, "
            f"transparency={self.config.noise_transparency}."
        )
        return add_noise_to_data(
            X_chunk,
            noise_type=self.config.noise_type,
            quantity=self.config.noise_quantity,
            random_state=self.config.random_state
        )
        # No esta modificando la transparencia
    
    
    def _split_data(
        self, 
        X_chunk: np.ndarray, 
        y_chunk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data chunk into training and testing sets.
    
        Args:
            X_chunk (np.ndarray): Input features.
            y_chunk (np.ndarray): Input labels.
    
        Returns:
            Tuple:
                X_train, X_test, y_train, y_test
        """
        self.logger.info(
            f"Splitting data: {self.config.train_split * 100:.1f}% train / "
            f"{self.config.test_split * 100:.1f}% test (random_state={self.config.random_state})"
        )
    
        # First split: train/test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_chunk,
            y_chunk,
            test_size=self.config.test_split,
            random_state=self.config.random_state,
            stratify=y_chunk
        )
    
        if self.config.vary_train_split:
            self.logger.info(
                "Varying training size after initial split: "
                f"{self.config.train_split * 100:.1f}% of training set."
            )
            X_train, _, y_train, _ = train_test_split(
                X_train_full,
                y_train_full,
                train_size=self.config.train_split,
                random_state=self.config.random_state,
                stratify=y_train_full
            )
        else:
            self.logger.info(
                "Using full training data without second split."
            )
            X_train, y_train = X_train_full, y_train_full
    
        self.logger.info(
            f"Data split complete. Train size: {X_train.shape}, Test size: {X_test.shape}"
        )
    
        return X_train, X_test, y_train, y_test
    
        
    
    def _reshape_images(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reshapes flattened input arrays into square image format.
    
        Args:
            X_train (np.ndarray): Flattened training data.
            X_test (np.ndarray): Flattened testing data.
    
        Returns:
            Tuple containing:
                - train_images (np.ndarray): Reshaped training images.
                - test_images (np.ndarray): Reshaped testing images.
        """
        self.logger.info("Reshaping train and test data into square images...")
    
        train_images = reshape_to_square_image(X_train)
        test_images = reshape_to_square_image(X_test)
    
        self.logger.info(
            f"Reshaping complete. "
            f"Train images shape: {train_images.shape}, "
            f"Test images shape: {test_images.shape}"
        )
    
        return train_images, test_images


    
##########################################
# 1. Orchestration
##########################################
    
    def run_experiment(self) -> None:
        """
        Manages the execution of one experiment using the MNIST dataset.

        Responsibilities:
        - Reads data in chunks
        - Process data and generates views.
        - Trains and tests the models.
        - Saves results.
        - Manages memory cleanup.

        Note:
            Multiple instances can run with different parameters.
        """
        start_time = time.perf_counter()
        
        self.logger.info(
            f"Reading dataset from {self.config.file_path} "
            f"(chunk size: {self.config.chunk_size})."
        )
        
        for X_chunk, y_chunk in tqdm(
            read_csv_in_chunks(
                self.config.file_path, 
                self.config.chunk_size), desc="Chunks:"
        ):
            # Process, train, and test the model on each chunk
            self.prepare_multiview_data_chunk(X_chunk, y_chunk)

        filename = generate_filename(
            path=Path(self.config.results_dir),
            name=self.config.exp_name,
            format="json"
        )
        self.save_results_to_json(str(filename))
        self.logger.info(f"Results successfully saved to {filename}")
        

        self.logger.info("Forced Garbage collection executed to Free Memory.")
        gc.collect()

        end_time = time.perf_counter()

        def format_elapsed_time(seconds: float) -> str:
            """Formats time for better readability."""
            minutes = seconds / 60
            if minutes < 60:
                return f"{minutes:.2f} minutes"
            hours = minutes / 60
            return f"{hours:.2f} hours"
        
        
        elapsed_time = end_time - start_time
        self.logger.info(
            f"========== [Experiment Completed] "
            f"in {format_elapsed_time(elapsed_time)} =========="
        )