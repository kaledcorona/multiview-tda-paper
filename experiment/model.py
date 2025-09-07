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
from typing import Dict, Tuple, TypeAlias
from math import isqrt, ceil
import itertools
import random

from collections import defaultdict
# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  

# ============================
# Local Application Imports
# ============================
#from my_project.utils import helper_function
#from my_project.models import User
#from my_project.config import SETTINGS
from polystack import Polystack
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

def train_multiview_stacking(
    X_train: Dict[str, np.ndarray],
    X_test: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 123,
    n_jobs: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Multi-View Stacking model using the provided training and test data.

    This function trains a stacking model where multiple views (feature sets) 
    are combined to improve predictive performance.

    Args:
        X_train (Dict[str, np.ndarray]): A dictionary of training feature sets, 
            where keys are view names (e.g., "view1", "view2") and values are 
            NumPy arrays with shape (n_samples, n_features).
        X_test (Dict[str, np.ndarray]): A dictionary of test feature sets, 
            similar to `X_train`.
        y_train (np.ndarray): Training labels, shape (n_samples,).
        y_test (np.ndarray): Test labels, shape (n_samples,).
        n_estimators (int, optional): Number of estimators for the Random Forest 
            meta-learner. Default is 50.
        random_state (int, optional): Random state for reproducibility. Default is 123.
        n_jobs (int, optional): Number of parallel jobs for the Random Forest. 
            Default is 3.

    Returns:
        Tuple[Dict[str, float], np.ndarray, np.ndarray]:
            - A classification report as a dictionary.
            - Predicted labels for the test set.
            - True labels for the test set.

    Example:
        >>> X_train = {"view1": np.random.rand(100, 10), "view2": np.random.rand(100, 15)}
        >>> X_test = {"view1": np.random.rand(50, 10), "view2": np.random.rand(50, 15)}
        >>> y_train = np.random.randint(0, 2, 100)
        >>> y_test = np.random.randint(0, 2, 50)
        >>> report, y_pred, y_test = train_multiview_stacking(X_train, X_test, y_train, y_test)
        >>> print(report)
    """
    # Validate inputs
    if not X_train or not X_test:
        raise ValueError("X_train and X_test cannot be empty.")
    if set(X_train.keys()) != set(X_test.keys()):
        raise ValueError("X_train and X_test must have the same keys (views).")

    # Initialize meta-learner
    meta_learner = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs
    )


    min_class_size = min(np.bincount(y_train))  # Find smallest class size
    n_splits = min(5, min_class_size)
    # Check this, I think it always chooose 5
    
    # Create and train the MultiView stacking model
    multiview_model = Polystack(final_estimator=meta_learner,
                                random_state=random_state,
                                cv=n_splits,
                                n_jobs=n_jobs)

    
    multiview_model.fit(X_train, y_train)

    
    y_pred : np.ndarray = multiview_model.predict(X_test)
    
    
    return (y_pred, y_test)

