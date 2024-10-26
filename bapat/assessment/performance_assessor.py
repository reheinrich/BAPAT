"""
performance_assessor.py

Defines the PerformanceAssessor class that orchestrates the calculation and plotting of performance metrics.
"""

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from bapat.assessment import metrics
from bapat.assessment import plotting


class PerformanceAssessor:
    """
    This class provides methods to evaluate the performance of a classification model
    by computing several key metrics and generating plots.
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        classes: Optional[Tuple[str, ...]] = None,
        task: Literal["binary", "multilabel"] = "multilabel",
        metrics_list: Tuple[str, ...] = (
            "recall",
            "precision",
            "f1",
            "ap",
            "auroc",
            "accuracy",
        ),
    ) -> None:
        """
        Initialize the PerformanceAssessor.

        Args:
            num_classes (int): The number of classes in the classification problem.
            threshold (float): The threshold for considering a label as positive.
            classes (Optional[Tuple[str, ...]]): Optional tuple of class names.
            task (Literal["binary", "multilabel"]): The type of classification task.
            metrics_list (Tuple[str, ...]): Tuple of metric names to compute.
        """
        # Validate inputs
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        if not isinstance(threshold, float) or not 0 < threshold < 1:
            raise ValueError("threshold must be a float between 0 and 1 (exclusive).")

        if classes is not None:
            if not isinstance(classes, tuple):
                raise ValueError("classes must be a tuple of strings.")
            if len(classes) != num_classes:
                raise ValueError(
                    f"Length of classes ({len(classes)}) must match num_classes ({num_classes})."
                )
            if not all(isinstance(class_name, str) for class_name in classes):
                raise ValueError("All elements in classes must be strings.")

        valid_metrics = {"accuracy", "recall", "precision", "f1", "ap", "auroc"}
        if not all(metric in valid_metrics for metric in metrics_list):
            raise ValueError(
                f"Some metrics in {metrics_list} are not supported. Valid options are {valid_metrics}."
            )

        self.num_classes = num_classes
        self.threshold = threshold
        self.classes = classes
        self.task = task
        self.metrics_list = metrics_list

        # Default colors for plotting
        self.colors = ["#3A50B1", "#61A83E", "#D74C4C", "#A13FA1", "#D9A544", "#F3A6E0"]

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate multiple performance metrics for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a NumPy array.
            labels (np.ndarray): Ground truth labels as a NumPy array.
            per_class_metrics (bool): If True, returns metrics for each class individually.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metrics.
        """
        # Determine the averaging method
        if per_class_metrics and self.num_classes == 1:
            averaging_method = "macro"
        else:
            averaging_method = None if per_class_metrics else "macro"

        metrics_results = {}

        for metric_name in self.metrics_list:
            if metric_name == "recall":
                result = metrics.calculate_recall(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["Recall"] = np.atleast_1d(result)
            elif metric_name == "precision":
                result = metrics.calculate_precision(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["Precision"] = np.atleast_1d(result)
            elif metric_name == "f1":
                result = metrics.calculate_f1_score(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["F1"] = np.atleast_1d(result)
            elif metric_name == "ap":
                result = metrics.calculate_average_precision(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    averaging_method=averaging_method,
                )
                metrics_results["AP"] = np.atleast_1d(result)
            elif metric_name == "auroc":
                result = metrics.calculate_auroc(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    averaging_method=averaging_method,
                )
                metrics_results["AUROC"] = np.atleast_1d(result)
            elif metric_name == "accuracy":
                result = metrics.calculate_accuracy(
                    predictions=predictions,
                    labels=labels,
                    task=self.task,
                    num_classes=self.num_classes,
                    threshold=self.threshold,
                    averaging_method=averaging_method,
                )
                metrics_results["Accuracy"] = np.atleast_1d(result)

        if per_class_metrics:
            columns = (
                self.classes
                if self.classes
                else [f"Class {i}" for i in range(self.num_classes)]
            )
        else:
            columns = ["Overall"]

        metrics_data = {key: np.atleast_1d(value) for key, value in metrics_results.items()}
        return pd.DataFrame.from_dict(metrics_data, orient="index", columns=columns)

    def plot_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ) -> None:
        """
        Plot performance metrics for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a NumPy array.
            labels (np.ndarray): Ground truth labels as a NumPy array.
            per_class_metrics (bool): If True, plots metrics for each class individually.
            cmap (str): Name of the colormap to be used.
        """
        metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics)

        if per_class_metrics:
            plotting.plot_metrics_per_class(metrics_df, self.colors)
        else:
            plotting.plot_overall_metrics(metrics_df, self.colors)

    def plot_metrics_all_thresholds(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        per_class_metrics: bool = False,
    ) -> None:
        """
        Plot performance metrics across thresholds for the given predictions and labels.

        Args:
            predictions (np.ndarray): Model output predictions as a NumPy array.
            labels (np.ndarray): Ground truth labels as a NumPy array.
            per_class_metrics (bool): If True, plots metrics for each class individually.
            cmap (str): Name of the colormap to be used.
        """
        original_threshold = self.threshold
        thresholds = np.arange(0.05, 1.0, 0.05)
        metrics_to_plot = [m for m in self.metrics_list if m not in ["auroc", "ap"]]

        if per_class_metrics:
            class_names = (
                self.classes
                if self.classes
                else [f"Class {i}" for i in range(self.num_classes)]
            )
            metric_values_dict_per_class = {
                class_name: {metric: [] for metric in metrics_to_plot}
                for class_name in class_names
            }

            for thresh in thresholds:
                self.threshold = thresh
                metrics_df = self.calculate_metrics(
                    predictions, labels, per_class_metrics=True
                )
                for metric_name in metrics_to_plot:
                    metric_label = (
                        metric_name.capitalize() if metric_name != "f1" else "F1"
                    )
                    for class_name in class_names:
                        value = metrics_df.loc[metric_label, class_name]
                        metric_values_dict_per_class[class_name][metric_name].append(
                            value
                        )

            self.threshold = original_threshold

            plotting.plot_metrics_across_thresholds_per_class(
                thresholds,
                metric_values_dict_per_class,
                metrics_to_plot,
                class_names,
                self.colors,
            )
        else:
            metric_values_dict = {metric_name: [] for metric_name in metrics_to_plot}

            for thresh in thresholds:
                self.threshold = thresh
                metrics_df = self.calculate_metrics(
                    predictions, labels, per_class_metrics=False
                )
                for metric_name in metrics_to_plot:
                    metric_label = (
                        metric_name.capitalize() if metric_name != "f1" else "F1"
                    )
                    value = metrics_df.loc[metric_label, "Overall"]
                    metric_values_dict[metric_name].append(value)

            self.threshold = original_threshold

            plotting.plot_metrics_across_thresholds(
                thresholds,
                metric_values_dict,
                metrics_to_plot,
                self.colors,
            )

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Plot confusion matrices for each class using scikit-learn's ConfusionMatrixDisplay.

        Args:
            predictions (np.ndarray): Model output predictions as a NumPy array.
            labels (np.ndarray): Ground truth labels as a NumPy array.
        """

        if self.task == "binary":
            # Apply threshold to get predicted class labels
            y_pred = (predictions >= self.threshold).astype(int).flatten()
            y_true = labels.flatten()
            # Compute normalized confusion matrix
            conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            conf_mat = np.round(conf_mat, 2)  # Round to 2 decimals
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_mat, display_labels=["Negative", "Positive"]
            )
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(
                cmap="Reds", ax=ax, colorbar=False, values_format=".2f"
            )  # Use .2f to show 2 decimal places
            ax.set_title("Confusion Matrix")
            plt.show()

        elif self.task == "multilabel":
            # Apply threshold to get predicted class labels
            y_pred = (predictions >= self.threshold).astype(int)
            y_true = labels
            # Compute confusion matrices for each class
            conf_mats = []
            class_names = (
                self.classes
                if self.classes
                else [f"Class {i}" for i in range(self.num_classes)]
            )
            for i in range(self.num_classes):
                conf_mat = confusion_matrix(
                    y_true[:, i], y_pred[:, i], normalize="true"
                )
                conf_mat = np.round(conf_mat, 2)  # Round to 2 decimals
                conf_mats.append(conf_mat)
            # Determine the number of rows and columns for subplots
            num_matrices = self.num_classes
            n_cols = int(np.ceil(np.sqrt(num_matrices)))
            n_rows = int(np.ceil(num_matrices / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = axes.flatten()
            for idx, (conf_mat, class_name) in enumerate(zip(conf_mats, class_names)):
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=conf_mat, display_labels=["Negative", "Positive"]
                )
                disp.plot(
                    cmap="Reds", ax=axes[idx], colorbar=False, values_format=".2f"
                )  # .2f for 2 decimal places
                axes[idx].set_title(f"{class_name}")
                axes[idx].set_xlabel("Predicted class")
                axes[idx].set_ylabel("True class")
            # Remove any unused axes
            for ax in axes[num_matrices:]:
                fig.delaxes(ax)
            plt.tight_layout()
            plt.show()

        else:
            raise ValueError(f"Unsupported task type: {self.task}")
