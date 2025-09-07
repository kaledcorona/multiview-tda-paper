"""
Module Description:
-------------------
Visualization module

Author: Kaled Corona
Date: 2025-04-01
"""

# ============================
# Standard Library Imports
# ============================
from typing import List, Tuple, Dict, Any

# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ============================
# Local Application Imports
# ============================

# ============================
# Configuration & Settings
# ============================


# ============================
# Main Execution 
# ============================


def plot_violin_f1(
    f1_scores: list[float],
    model_name: str = "Model",
    title: str = "F1 Score Distribution",
    figsize: tuple = (5, 6),
    inner: str = "quartile",
    palette: str = "muted",
    show: bool = True
) -> None:
    """
    Plots a violin plot for the given list of F1 scores.

    Args:
        f1_scores (List[float]): List of F1 scores.
        model_name (str): Label for the model/category.
        title (str): Plot title.
        figsize (tuple): Size of the figure (width, height).
        inner (str): Interior style for the violin ("box", "quartile", "point", or None).
        palette (str): Seaborn color palette name.
        show (bool): Whether to immediately display the plot (plt.show()).
    """
    df = pd.DataFrame({
        "model": [model_name] * len(f1_scores),
        "f1": f1_scores
    })

    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x="model", y="f1", hue="model", legend=False, inner=inner, palette=palette)
    plt.title(title)
    plt.ylabel("F1 Score")
    plt.tight_layout()

    if show:
        plt.show()



def get_metric_arrays(summary: Dict[int, Any], split_percentages: List[int], metric: str) -> Tuple[np.ndarray, ...]:
    """Extract mean and std arrays for TDA and non-TDA models for a given metric."""
    tda_means, tda_stds, non_tda_means, non_tda_stds = [], [], [], []

    for sp in split_percentages:
        tda = summary[sp]["tda_metrics"]
        non_tda = summary[sp]["non_tda_metrics"]

        tda_means.append(tda["mean_metrics"].get(metric, np.nan))
        tda_stds.append(tda["std_metrics"].get(metric, 0))
        non_tda_means.append(non_tda["mean_metrics"].get(metric, np.nan))
        non_tda_stds.append(non_tda["std_metrics"].get(metric, 0))

    return (np.array(tda_means), np.array(tda_stds),
            np.array(non_tda_means), np.array(non_tda_stds))


def plot_single_metric(ax, x_vals: List[int],
                       tda: Tuple[np.ndarray, np.ndarray],
                       non_tda: Tuple[np.ndarray, np.ndarray],
                       title: str) -> None:
    """Plots one metric subplot."""
    tda_means, tda_stds = tda
    non_means, non_stds = non_tda

    ax.errorbar(x_vals, tda_means, yerr=tda_stds, fmt='-o', label='MVStacking-TDA')
    ax.errorbar(x_vals, non_means, yerr=non_stds, fmt='-s', label='Raw data')
    ax.set_title(title)
    ax.set_xlabel("Training Split (%)")
    ax.set_ylabel("Metric Value")
    ax.grid(True)
    ax.legend()


def setup_plot_grid(title : str) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Initializes the 2x2 plot grid."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(title, fontsize=14)
    return fig, axes.flatten()


def plot_tda_comparison(summary: Dict[int, Any], title_plot : str) -> None:
    """Main function to create the comparison plot."""
    split_percentages = sorted(summary.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    titles = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = setup_plot_grid(title_plot)

    for ax, metric, title in zip(axes, metrics, titles):
        tda_vals = get_metric_arrays(summary, split_percentages, metric)[:2]
        non_tda_vals = get_metric_arrays(summary, split_percentages, metric)[2:]
        plot_single_metric(ax, split_percentages, tda_vals, non_tda_vals, title)

    plt.tight_layout()
    plt.savefig('MVStacking-tda-comparasion.png', dpi=300, bbox_inches='tight')
    plt.show()





#//////////////////////////// old code



import numpy as np
import matplotlib.pyplot as plt

def plot_tda_comparison_old(summary):
    """
    Plots four subplots comparing mean metrics (Accuracy, Precision, Recall, F1-Score)
    for TDA vs. non-TDA models, including error bars for standard deviations.

    :param summary: Dictionary returned by analyze_experiments, of the form:
      {
        train_split_percentage: {
          "files": [...],
          "tda_metrics": {
            "mean_metrics": { ... },
            "std_metrics": { ... }
          },
          "non_tda_metrics": {
            "mean_metrics": { ... },
            "std_metrics": { ... }
          }
        },
        ...
      }
    """

    # Sort the training split percentages
    split_percentages = sorted(summary.keys())

    # Prepare figure and axes for four metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Raw data with salt noise vs MVStacking-TDA across metrics", fontsize=14)
    axes = axes.flatten()  # Flatten to iterate easily

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    titles = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Helper function to get TDA / non-TDA metric arrays
    def get_metric_arrays(metric):
        tda_means, tda_stds = [], []
        non_tda_means, non_tda_stds = [], []

        for sp in split_percentages:
            # Extract TDA metrics
            tda_mean = summary[sp]["tda_metrics"].get("mean_metrics", {}).get(metric)
            tda_std = summary[sp]["tda_metrics"].get("std_metrics", {}).get(metric)
            if tda_mean is not None:
                tda_means.append(tda_mean)
                tda_stds.append(tda_std)
            else:
                # If no TDA data, append NaN for mean and 0 for std
                tda_means.append(np.nan)
                tda_stds.append(0)

            # Extract non-TDA metrics
            non_tda_mean = summary[sp]["non_tda_metrics"].get("mean_metrics", {}).get(metric)
            non_tda_std = summary[sp]["non_tda_metrics"].get("std_metrics", {}).get(metric)
            if non_tda_mean is not None:
                non_tda_means.append(non_tda_mean)
                non_tda_stds.append(non_tda_std)
            else:
                # If no non-TDA data, append NaN for mean and 0 for std
                non_tda_means.append(np.nan)
                non_tda_stds.append(0)

        return (np.array(tda_means), np.array(tda_stds),
                np.array(non_tda_means), np.array(non_tda_stds))

    # Generate each subplot
    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]
        (tda_means, tda_stds, non_tda_means, non_tda_stds) = get_metric_arrays(metric)

        # Plot TDA
        ax.errorbar(split_percentages, tda_means, yerr=tda_stds, fmt='-o',
                    label='MVStacking-TDA')
        # Plot non-TDA
        ax.errorbar(split_percentages, non_tda_means, yerr=non_tda_stds, fmt='-s',
                    label='Raw')

        ax.set_title(title)
        ax.set_xlabel("Training Split (%)")
        ax.set_ylabel("Metric Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('MVStacking-tda-comparasion.png', dpi=300, bbox_inches='tight')
    plt.show()


