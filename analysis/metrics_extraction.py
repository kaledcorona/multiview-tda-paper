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
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean, stdev, median
import random


# ============================
# Third-Party Library Imports
# ============================
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================
# Local Application Imports
# ============================

# ============================
# Configuration & Settings
# ============================


# ============================
# Main Execution 
# ============================

@dataclass(frozen=True)
class ModelResult:
    model_id: int
    model_name: str
    views: set[str]
    
    predictions: list[float]
    ground_truth: list[float]
    random_state: int
    train_size: float
    test_size: float
    
    f1_score: float
    accuracy: float


@dataclass(frozen=True)
class ModelGroup:
    view_key: str
    results: list[ModelResult]
    views: set[str] = field(init=False)

    def __post_init__(self):
        if not self.results:
            raise ValueError("ModelGroup must contain at least one result.")
        view_set = self.results[0].views
        assert all(r.views == view_set for r in self.results), "Inconsistent views in ModelGroup"
        object.__setattr__(self, "views", view_set)
   

    def f1_scores(self) -> list[float]:
        return [r.f1_score for r in self.results]

    def average_f1(self) -> float:
        return mean(self.f1_scores())

    def std_f1(self) -> float:
        scores = self.f1_scores()
        return stdev(scores) if len(scores) > 1 else 0.0

    def median_f1(self) -> float:
        return median(self.f1_scores())

    def acc_scores(self) -> list[float]:
        return [r.accuracy for r in self.results]

    def average_accuracy(self) -> float:
        return mean(self.accuracy_scores())

    def std_accuracy(self) -> float:
        scores = self.accuracy_scores()
        return stdev(scores) if len(scores) > 1 else 0.0

    def median_accuracy(self) -> float:
        return median(self.accuracy_scores())


@dataclass(frozen=True)
class SplitGroup:
    train_size: float
    model_groups: dict[str, ModelGroup]

    def all_models(self) -> list[ModelResult]:
        return [r for mg in self.model_groups.values() for r in mg.results]


@dataclass(frozen=True)
class GroupedResults:
    splits: dict[float, SplitGroup]

    def get_split(self, train_size: float) -> SplitGroup | None:
        return self.splits.get(round(train_size, 2))


def load_json_file(file_path: Path) -> dict:
    """Loads a JSON file."""
    with file_path.open("r",encoding='utf-8') as file:
        return json.load(file)


def compute_f1(preds: List[int], gts: List[int]) -> float:
    # assume preds and gts are same length and nonempty
    return f1_score(gts, preds, average='macro', zero_division=0)


def parse_model_result(entry: dict) -> ModelResult:
    """Transform a raw dictionary into a ModelResult"""
    preds = entry["predictions"]
    truths = entry["ground_truth"]
    return ModelResult( 
        model_id=random.randint(100000,999999),
        model_name=entry["model_name"],
        views=set(entry["model_name"].split("-")),
        predictions=preds,
        ground_truth=truths,
        random_state=entry["random_state"],
        train_size=entry["train_size"],
        test_size=entry["test_size"],
        f1_score=compute_f1(preds,truths),
        accuracy=entry.get("accuracy", accuracy_score(truths, preds)),
    )


def read_json_files(directory: str) -> list[ModelResult]:
    """
    Reads JSON files in a directory and parses data.
    """
    path = Path(directory)
    results: list[ModelResults] = []

    return [
        parse_model_result(result)
        for file_path in path.glob("*.json")
        for result in load_json_file(file_path).get("results", [])
    ]


def build_grouped_results(models: list[ModelResult]) -> GroupedResults:
    split_dict: dict[float, dict[str, list[ModelResult]]] = defaultdict(lambda: defaultdict(list))

    # Outer key: train_size -> Inner: views_key -> List of ModelResult
    split_dict: dict[float, dict[str, list[ModelResult]]] = defaultdict(lambda: defaultdict(list))


    for model in models:
        split_key = round(model.train_size, 2)
        views_key = "-".join(sorted(model.views))
        split_dict[split_key][views_key].append(model)

    splits = {
        split_key: SplitGroup(
            train_size=split_key,
            model_groups={
                views_key: ModelGroup(view_key=views_key, results=res_list)
                for views_key, res_list in model_dict.items()
            }
        )
        for split_key, model_dict in split_dict.items()
    }

    return GroupedResults(splits=splits)





######################################




from matplotlib.lines import Line2D


def plot_aggregated_f1(
    json_paths: List[Path],
    experiment: int,
    title: str = None
) -> None:
    """
    Aggregates 1–25 experiment repeats and plots mean±var F1 per model
    with an aesthetic, minimalist style.
    """
    exp_idx = (experiment - 1) % 25
    scores: dict[str, List[float]] = {}
    for i, jp in enumerate(json_paths):
        if (i % 25) == exp_idx:
            data = load_json_file(jp)
            for mdl, preds, gts in extract_results(data):
                if len(preds)==len(gts) and preds:
                    scores.setdefault(mdl, []).append(compute_f1(preds, gts))

    if not scores:
        print(f"No data for experiment {experiment}.")
        return

    models   = list(scores)
    means    = np.array([np.mean(scores[m]) for m in models])
    variances= np.array([np.var (scores[m]) for m in models])
    order    = np.argsort(means)[::-1]
    models   = [models[i] for i in order]
    means    = means[order]
    variances= variances[order]

    # Colors & style
    colors = ['tomato' if 'tda' in m.lower() else 'steelblue' for m in models]
    mpl.rcParams.update({
        'font.family':     'sans-serif',
        'font.size':        10,
        'axes.facecolor':  '#f2f2f2',
        'grid.color':      'white',
        'grid.linestyle':  '-',
        'grid.linewidth':   1,
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axisbelow(True)
    bars = ax.bar(models, means,
                  yerr=variances, capsize=5,
                  color=colors,
                  edgecolor='black', linewidth=0.8)

    # Annotate
    for i, m in enumerate(means):
        ax.text(i, m + variances.max() + 0.001,
                f"{m:.4f}",
                ha='center', va='bottom',
                rotation=90, fontsize=8)

    # ΔF1 arrow
    vmin, vmax = means.min(), means.max()
    xpos = len(models) - 0.5
    ax.annotate(
        '',
        xy=(xpos, vmax),
        xytext=(xpos, vmin),
        arrowprops=dict(arrowstyle='<->', color='magenta', lw=2)
    )
    delta = vmax - vmin
    ax.text(xpos + 0.3,
            (vmin + vmax) / 2,
            f"ΔF1 = {delta:.4f}",
            color='magenta', va='center',
            fontweight='bold', fontsize=10)

    # Spines & grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')

    # Labels & title
    ax.set_xticklabels(models, rotation=90, fontsize=9)
    ax.set_ylabel("Mean F1 Score", fontsize=12)
    low = vmin - variances.max() - 0.01
    high= vmax + variances.max() + 0.01
    ax.set_ylim(low, high)
    ax.set_title(title or f"Experiment {experiment}: Mean±Var F1 per Model", fontsize=14)

    # Legend with subtle frame
    tda_patch    = mpatches.Patch(color='tomato',    label='With TDA')
    non_tda_patch= mpatches.Patch(color='steelblue', label='Without TDA')
    arrow_line   = Line2D([0], [0], color='magenta', lw=2, label='ΔF1 (max−min)')
    leg = ax.legend(handles=[tda_patch, non_tda_patch, arrow_line], loc='upper right')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#888888')
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.show()



def show_metrics_per_model(json_path: Union[str, Path]) -> None:
    """
    Loads predictions from a JSON file and prints metrics per model.
    """
    data = load_json_file(json_path)
    results = extract_results(data)

    for model_name, preds, gts in results:
        metrics = compute_metrics(preds, gts)
        if metrics:
            acc, prec, rec, f1 = metrics
            print(f"Model: {model_name}")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1 Score : {f1:.4f}\n")
        else:
            print(f"Model: {model_name} - Invalid predictions or ground truths.\n")



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

def plot_f1_scores(json_path: Union[str, Path], title: str = "F1 Score per Model (Sorted & Highlighted)") -> None:
    """
    Plots F1 scores per model, highlights 'tda' models, sorts bars by score,
    adds annotations, shows min/max lines, highlights F1 score difference.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.
        title (str): Title of the plot.
    """
    data = load_json_file(json_path)
    results = extract_results(data)

    model_names = []
    f1_scores = []

    for model_name, preds, gts in results:
        metrics = compute_metrics(preds, gts)
        if metrics:
            _, _, _, f1 = metrics
            model_names.append(model_name)
            f1_scores.append(f1)

    if not model_names:
        print("No valid models found.")
        return

    # Identify colors based on presence of 'tda'
    bar_colors = ['tomato' if 'tda' in name.lower() else 'steelblue' for name in model_names]

    # Sort data by F1 score
    sorted_data = sorted(zip(f1_scores, model_names, bar_colors), reverse=True)
    f1_scores, model_names, bar_colors = zip(*sorted_data)

    # Aesthetic refinements
    mpl.rcParams.update({'font.family': 'sans-serif'})
    plt.figure(figsize=(14, 8))

    # Bar plot
    bars = plt.bar(model_names, f1_scores, color=bar_colors)

    # Annotate values
    for i, score in enumerate(f1_scores):
        plt.text(i, score + 0.0005, f"{score:.4f}", ha='center', va='bottom', fontsize=8, rotation=90)

    # Min/Max reference lines
    min_f1 = min(f1_scores)
    max_f1 = max(f1_scores)
    y_min = max(0.0, min_f1 - 0.01)
    plt.axhline(min_f1, color='gray', linestyle='--', linewidth=1)
    plt.axhline(max_f1, color='black', linestyle='--', linewidth=1)

    # Draw vertical line for the F1 score difference
    x_position = len(model_names) - 0.5  # At the far right, slight offset
    plt.plot([x_position, x_position], [min_f1, max_f1], color='magenta', linestyle='-', linewidth=2)

    # Add a label for the height difference
    f1_difference = max_f1 - min_f1
    plt.text(x_position + 0.5, (min_f1 + max_f1) / 2,
             f"ΔF1 = {f1_difference:.4f}",
             ha='left', va='center', color='magenta', fontsize=10, fontweight='bold', rotation=90)

    # Labels & layout
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim([y_min, 1.01])
    plt.xticks(rotation=90, fontsize=9)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    # Custom legend
    tda_patch = mpatches.Patch(color='tomato', label='Models with TDA')
    non_tda_patch = mpatches.Patch(color='steelblue', label='Models without TDA')
    min_patch = mpatches.Patch(color='gray', label=f"Min F1 = {min_f1:.4f}")
    max_patch = mpatches.Patch(color='black', label=f"Max F1 = {max_f1:.4f}")

    plt.legend(handles=[tda_patch, non_tda_patch, min_patch, max_patch], loc='upper right')

    plt.tight_layout()
    plt.show()

######################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from pathlib import Path
from ipywidgets import interact
from typing import List, Union

def interactive_f1_plot(json_paths: List[Union[str, Path]], title_base: str = "F1 Score Plot") -> None:
    """
    Creates an interactive slider to plot F1 scores from multiple JSON files.

    Args:
        json_paths (List[Union[str, Path]]): List of JSON file paths.
        title_base (str): Base title for the plots.
    """
    def plot_for_index(index: int):
        if index >= len(json_paths):
            print("Index out of bounds.")
            return

        json_path = Path(json_paths[index])
        data = load_json_file(json_path)
        results = extract_results(data)

        model_names = []
        f1_scores = []

        for model_name, preds, gts in results:
            metrics = compute_metrics(preds, gts)
            if metrics:
                _, _, _, f1 = metrics
                model_names.append(model_name)
                f1_scores.append(f1)

        if not model_names:
            print("No valid models found.")
            return

        # Identify colors
        bar_colors = ['tomato' if 'tda' in name.lower() else 'steelblue' for name in model_names]

        # Sort
        sorted_data = sorted(zip(f1_scores, model_names, bar_colors), reverse=True)
        f1_scores, model_names, bar_colors = zip(*sorted_data)

        # Plot
        mpl.rcParams.update({'font.family': 'sans-serif'})
        plt.figure(figsize=(14, 8))
        bars = plt.bar(model_names, f1_scores, color=bar_colors)

        for i, score in enumerate(f1_scores):
            plt.text(i, score + 0.0005, f"{score:.4f}", ha='center', va='bottom', fontsize=8, rotation=90)

        min_f1 = min(f1_scores)
        max_f1 = max(f1_scores)
        y_min = max(0.0, min_f1 - 0.01)

        plt.axhline(min_f1, color='gray', linestyle='--', linewidth=1)
        plt.axhline(max_f1, color='black', linestyle='--', linewidth=1)

        # Vertical difference line
        x_position = len(model_names) - 0.5
        plt.plot([x_position, x_position], [min_f1, max_f1], color='magenta', linestyle='-', linewidth=2)

        # Label difference
        f1_difference = max_f1 - min_f1
        plt.text(x_position + 0.5, (min_f1 + max_f1) / 2,
                 f"ΔF1 = {f1_difference:.4f}",
                 ha='left', va='center', color='magenta', fontsize=10, fontweight='bold', rotation=90)

        # Labels
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title(f"{title_base} (Index {index})", fontsize=14)
        plt.ylim([y_min, 1.01])
        plt.xticks(rotation=90, fontsize=9)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)

        # Legend
        tda_patch = mpatches.Patch(color='tomato', label='Models with TDA')
        non_tda_patch = mpatches.Patch(color='steelblue', label='Models without TDA')
        min_patch = mpatches.Patch(color='gray', label=f"Min F1 = {min_f1:.4f}")
        max_patch = mpatches.Patch(color='black', label=f"Max F1 = {max_f1:.4f}")
        plt.legend(handles=[tda_patch, non_tda_patch, min_patch, max_patch], loc='upper right')

        plt.tight_layout()
        plt.show()

    # Create interactive slider
    interact(plot_for_index, index=list(range(0, len(json_paths), 2)))


###########################


    




def aggregate_metrics(metrics: List[Tuple[float, float, float, float]]) -> dict:
    if not metrics:
        return {"mean_metrics": {}, "std_metrics": {}}
    arr = np.array(metrics)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    keys = ["accuracy", "precision", "recall", "f1_score"]
    return {
        "mean_metrics": dict(zip(keys, map(float, mean))),
        "std_metrics": dict(zip(keys, map(float, std))),
    }


def analyze_experiments(directory: str) -> Dict[int, Any]:
    split_groups = group_by_split(directory)
    summary = {}

    for split, info in split_groups.items():
        summary[split] = {
            "files": info["files"],
            "tda_metrics": aggregate_metrics(info["tda_metrics"]),
            "non_tda_metrics": aggregate_metrics(info["non_tda_metrics"])
        }

    return summary


from scipy.stats import ttest_ind, mannwhitneyu
from typing import Literal

def perform_statistical_test(
    groups: Dict[int, dict],
    metric: Literal["accuracy", "precision", "recall", "f1_score"] = "f1_score",
    test_type: Literal["t-test", "mannwhitney"] = "t-test",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
) -> Tuple[float, float]:
    """
    Perform a statistical test between TDA and non-TDA model metrics.
    
    Args:
        groups: Output from `group_by_split`, mapping splits to metric data.
        metric: Metric to compare ("accuracy", "precision", "recall", "f1_score").
        test_type: Type of test ("t-test" or "mannwhitney").
        alternative: Type of alternative hypothesis.

    Returns:
        Tuple containing the test statistic and p-value.
    """
    tda_vals = []
    non_tda_vals = []

    for split_data in groups.values():
        for tda in split_data["tda_metrics"]:
            tda_vals.append(tda[["accuracy", "precision", "recall", "f1_score"].index(metric)])
        for non_tda in split_data["non_tda_metrics"]:
            non_tda_vals.append(non_tda[["accuracy", "precision", "recall", "f1_score"].index(metric)])

    if not tda_vals or not non_tda_vals:
        raise ValueError("Insufficient data for statistical testing.")

    if test_type == "t-test":
        stat, pval = ttest_ind(tda_vals, non_tda_vals, alternative=alternative, equal_var=False)
    elif test_type == "mannwhitney":
        stat, pval = mannwhitneyu(tda_vals, non_tda_vals, alternative=alternative)
    else:
        raise ValueError("Invalid test_type. Use 't-test' or 'mannwhitney'.")

    return stat, pval



