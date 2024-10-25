"""
plotting.py

Module containing functions to plot performance metrics.
"""

from typing import List, Literal
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
    # Extract the metric names and their values
    metrics = metrics_df.index  # Metric names
    values = metrics_df["Overall"].values  # Metric values

    # Plot the bar chart with specified colors
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=colors)

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
    metric_values_dict: dict,
    metrics_to_plot: List[str],
    colors: List[str],
) -> None:
    """
    Plots metrics across different thresholds.
    """
    # Define a list of line styles
    line_styles = ["-", "--", "-.", ":", (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    plt.figure(figsize=(10, 6))

    # Plot each metric across thresholds
    for i, metric_name in enumerate(metrics_to_plot):
        metric_values = metric_values_dict[metric_name]
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
    metric_values_dict_per_class: dict,
    metrics_to_plot: List[str],
    class_names: List[str],
    colors: List[str],
) -> None:
    """
    Plots metrics across different thresholds per class.
    """
    num_classes = len(class_names)

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
        ax = axes[class_idx]
        metric_values_dict = metric_values_dict_per_class[class_name]

        # Plot each metric for the current class
        for i, metric_name in enumerate(metrics_to_plot):
            metric_values = metric_values_dict[metric_name]
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
    if task == "binary":
        # For binary classification, conf_mat is a 2x2 matrix
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap="Reds", cbar=False)
        plt.title(f"{class_names[0]}")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.show()
    else:
        # For multilabel classification, conf_mat is (num_labels, 2, 2)
        num_labels = conf_mat.shape[0]

        # Determine grid size based on the number of classes
        n_cols = int(np.ceil(np.sqrt(num_labels)))
        n_rows = int(np.ceil(num_labels / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

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
