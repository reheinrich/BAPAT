# plotting.py

"""
Module containing functions to plot performance metrics.
"""

from typing import List, Dict, Literal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_overall_metrics(
    metrics_df: pd.DataFrame,
    colors: List[str],
) -> None:
    """
    Plots a bar chart for overall performance metrics.
    """
    # Input validation
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError("metrics_df must be a pandas DataFrame.")
    if 'Overall' not in metrics_df.columns:
        raise KeyError("metrics_df must contain an 'Overall' column.")
    if metrics_df.empty:
        raise ValueError("metrics_df is empty.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Extract the metric names and their values
    metrics = metrics_df.index  # Metric names
    values = metrics_df["Overall"].values  # Metric values

    # Plot the bar chart with specified colors
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=colors[:len(metrics)])

    # Add titles and labels
    plt.title("Overall Metric Scores", fontsize=16)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Add gridlines and adjust the layout
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.show()


def plot_metrics_per_class(
    metrics_df: pd.DataFrame,
    colors: List[str],
) -> None:
    """
    Plots metric values per class, with each metric represented by a distinct color and line.
    """
    # Input validation
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError("metrics_df must be a pandas DataFrame.")
    if metrics_df.empty:
        raise ValueError("metrics_df is empty.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Define a list of line styles
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    plt.figure(figsize=(10, 6))

    # Plot each metric for each class with a unique color
    for i, metric_name in enumerate(metrics_df.index):
        values = metrics_df.loc[metric_name]  # Metric values for each class
        classes = metrics_df.columns  # Class labels

        # Plot the metric line with markers
        plt.plot(
            classes,
            values,
            label=metric_name,
            marker="o",
            markersize=8,
            linewidth=2,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i % len(colors)],
        )

    # Add titles and labels
    plt.title("Metric Scores per Class", fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    # Show legend
    plt.legend(loc="lower right")

    # Add gridlines and adjust layout
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_metrics_across_thresholds(
    thresholds: np.ndarray,
    metric_values_dict: Dict[str, np.ndarray],
    metrics_to_plot: List[str],
    colors: List[str],
) -> None:
    """
    Plots metrics across different thresholds.
    """
    # Input validation
    if not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds must be a numpy ndarray.")
    if thresholds.size == 0:
        raise ValueError("thresholds array is empty.")
    if not isinstance(metric_values_dict, dict):
        raise TypeError("metric_values_dict must be a dictionary.")
    if not isinstance(metrics_to_plot, list):
        raise TypeError("metrics_to_plot must be a list.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Define a list of line styles
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    plt.figure(figsize=(10, 6))

    # Plot each metric across thresholds
    for i, metric_name in enumerate(metrics_to_plot):
        if metric_name not in metric_values_dict:
            raise KeyError(f"Metric '{metric_name}' not found in metric_values_dict.")
        metric_values = metric_values_dict[metric_name]
        if len(metric_values) != len(thresholds):
            raise ValueError(f"Length of metric '{metric_name}' values does not match length of thresholds.")
        line_style = line_styles[i % len(line_styles)]
        plt.plot(
            thresholds,
            metric_values,
            label=metric_name.capitalize() if metric_name != "f1" else "F1",
            linestyle=line_style,
            linewidth=2,
            color=colors[i % len(colors)],
        )

    plt.title("Metrics across Different Thresholds", fontsize=16)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics_across_thresholds_per_class(
    thresholds: np.ndarray,
    metric_values_dict_per_class: Dict[str, Dict[str, np.ndarray]],
    metrics_to_plot: List[str],
    class_names: List[str],
    colors: List[str],
) -> None:
    """
    Plots metrics across different thresholds per class.
    """
    # Input validation
    if not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds must be a numpy ndarray.")
    if thresholds.size == 0:
        raise ValueError("thresholds array is empty.")
    if not isinstance(metric_values_dict_per_class, dict):
        raise TypeError("metric_values_dict_per_class must be a dictionary.")
    if not isinstance(metrics_to_plot, list):
        raise TypeError("metrics_to_plot must be a list.")
    if not isinstance(class_names, list):
        raise TypeError("class_names must be a list.")
    if not isinstance(colors, list):
        raise TypeError("colors must be a list.")
    if len(colors) == 0:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("class_names list is empty.")

    # Determine grid size based on the number of classes
    n_cols = int(np.ceil(np.sqrt(num_classes)))
    n_rows = int(np.ceil(num_classes / n_cols))

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    # Flatten the axes array for easy indexing
    if num_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define a list of line styles
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    for class_idx, class_name in enumerate(class_names):
        if class_name not in metric_values_dict_per_class:
            raise KeyError(f"Class '{class_name}' not found in metric_values_dict_per_class.")
        ax = axes[class_idx]
        metric_values_dict = metric_values_dict_per_class[class_name]

        # Plot each metric for the current class
        for i, metric_name in enumerate(metrics_to_plot):
            if metric_name not in metric_values_dict:
                raise KeyError(f"Metric '{metric_name}' not found for class '{class_name}'.")
            metric_values = metric_values_dict[metric_name]
            if len(metric_values) != len(thresholds):
                raise ValueError(f"Length of metric '{metric_name}' values for class '{class_name}' does not match length of thresholds.")
            line_style = line_styles[i % len(line_styles)]
            ax.plot(
                thresholds,
                metric_values,
                label=metric_name.capitalize() if metric_name != "f1" else "F1",
                linestyle=line_style,
                linewidth=2,
                color=colors[i % len(colors)],
            )

        ax.set_title(f"{class_name}", fontsize=12)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Metric Score", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

    # Hide any unused subplots
    for j in range(num_classes, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    conf_mat: np.ndarray,
    task: Literal["binary", "multiclass", "multilabel"],
    class_names: List[str],
) -> None:
    """
    Plot confusion matrices for each class in a single figure with multiple subplots.
    """
    # Input validation
    if not isinstance(conf_mat, np.ndarray):
        raise TypeError("conf_mat must be a numpy ndarray.")
    if conf_mat.size == 0:
        raise ValueError("conf_mat is empty.")
    if not isinstance(task, str) or task not in ["binary", "multiclass", "multilabel"]:
        raise ValueError("Invalid task. Expected 'binary', 'multiclass', or 'multilabel'.")
    if not isinstance(class_names, list):
        raise TypeError("class_names must be a list.")
    if len(class_names) == 0:
        raise ValueError("class_names list is empty.")

    if task == "binary":
        if conf_mat.shape != (2, 2):
            raise ValueError("For binary task, conf_mat must be of shape (2, 2).")
        if len(class_names) != 2:
            raise ValueError("For binary task, class_names must have exactly two elements.")
        # For binary classification, conf_mat is a 2x2 matrix
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap="Reds", cbar=False)
        plt.title(f"Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.show()
    else:
        # For multilabel or multiclass classification
        num_labels = conf_mat.shape[0]
        if conf_mat.shape[1:] != (2, 2):
            raise ValueError("For multilabel or multiclass task, conf_mat must have shape (num_labels, 2, 2).")
        if len(class_names) != num_labels:
            raise ValueError("Length of class_names must match number of labels in conf_mat.")

        # Determine grid size based on the number of classes
        n_cols = int(np.ceil(np.sqrt(num_labels)))
        n_rows = int(np.ceil(num_labels / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

        # Flatten the axes array for easy indexing
        if num_labels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(num_labels):
            cm = conf_mat[i]
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds", cbar=False, ax=ax)
            ax.set_title(f"{class_names[i]}")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")

        # Hide any unused subplots
        for j in range(num_labels, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
