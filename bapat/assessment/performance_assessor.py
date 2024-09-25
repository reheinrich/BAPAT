from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)


class PerformanceAssessor:
    """
    This class provides methods to evaluate the performance of a classification model
    (binary, multiclass, or multilabel) by computing several key metrics including recall,
    precision, F1 score, average precision (AP), accuracy, and area under the receiver operating characteristic curve (AUROC).

    Args:
        num_classes (int): The number of classes in the classification problem.
        threshold (float): The threshold for considering a label as positive. Must be between 0 and 1 (exclusive). Defaults to 0.5.
        classes (Optional[Tuple[str, ...]]): Optional tuple of strings of class names.
                                           If provided, its length must match num_classes.
        task (str): The type of classification task. Can be 'binary', 'multiclass', or 'multilabel'.
        metrics (Tuple[str]): Tuple of metric names to compute. Valid options include 'accuracy', 'recall', 'precision', 'f1', 'ap', 'auroc'.
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        classes: Optional[Tuple[str, ...]] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "multilabel",
        metrics: Tuple[str, ...] = (
            "recall",
            "precision",
            "f1",
            "ap",
            "auroc",
            "accuracy",
        ),
    ) -> None:
        # Assert that num_classes is a positive integer
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "num_classes must be a positive integer."

        # Assert that threshold is a float between 0 and 1 (exclusive)
        assert (
            isinstance(threshold, float) and 0 < threshold < 1
        ), "threshold must be a float between 0 and 1 (exclusive)."

        # If classes are provided, check that it's a tuple and its length matches num_classes
        if classes is not None:
            assert isinstance(classes, tuple), "classes must be a tuple of strings."
            assert (
                len(classes) == num_classes
            ), f"Length of classes ({len(classes)}) must match num_classes ({num_classes})."
            assert all(
                isinstance(class_name, str) for class_name in classes
            ), "All elements in classes must be strings."

        # Ensure metrics are valid
        valid_metrics = {"accuracy", "recall", "precision", "f1", "ap", "auroc"}
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Metric '{metric}' is not supported. Valid options are {valid_metrics}."
                )

        self.num_classes = num_classes
        self.threshold = threshold
        self.classes = classes
        self.task = task
        self.metrics = metrics

        self.colors = ['#3A50B1', '#61A83E', '#D74C4C', '#A13FA1', '#D9A544', '#F3A6E0']

    def _calculate_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        averaging_method: Optional[
            Literal["micro", "macro", "weighted", "none"]
        ] = "macro",
    ) -> torch.Tensor:
        """
        Calculate accuracy for the given predictions and labels.

        Accuracy measures the proportion of correct predictions made by the model compared to the total predictions.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            averaging_method (Optional[Literal['micro', 'macro', 'weighted', 'none']]):
                Method for averaging accuracy across classes. Defaults to 'macro'.

        Returns:
            torch.Tensor: Computed accuracy score.
        """
        accuracy = Accuracy(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            average=averaging_method,
            threshold=self.threshold,
        )
        return accuracy(preds=predictions, target=labels)

    def _calculate_recall(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        averaging_method: Optional[
            Literal["micro", "macro", "weighted", "none"]
        ] = "macro",
    ) -> torch.Tensor:
        """
        Calculate recall for the given predictions and labels.

        Recall measures the proportion of correctly identified positive labels among all actual positive labels.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            averaging_method (Optional[Literal['micro', 'macro', 'weighted', 'none']]):
                Method for averaging the recall across classes. Defaults to 'macro'.

        Returns:
            torch.Tensor: Computed recall score.
        """
        recall = Recall(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            average=averaging_method,
            threshold=self.threshold,
        )
        return recall(preds=predictions, target=labels)

    def _calculate_precision(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        averaging_method: Optional[
            Literal["micro", "macro", "weighted", "none"]
        ] = "macro",
    ) -> torch.Tensor:
        """
        Calculate precision for the given predictions and labels.

        Precision measures the proportion of correctly predicted positive labels among all positive predictions.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            averaging_method (Optional[Literal['micro', 'macro', 'weighted', 'none']]):
                Method for averaging the precision across classes. Defaults to 'macro'.

        Returns:
            torch.Tensor: Computed precision score.
        """
        precision = Precision(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            average=averaging_method,
            threshold=self.threshold,
        )
        return precision(preds=predictions, target=labels)

    def _calculate_f1_score(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        averaging_method: Optional[
            Literal["micro", "macro", "weighted", "none"]
        ] = "macro",
    ) -> torch.Tensor:
        """
        Calculate the F1 score for the given predictions and labels.

        The F1 score is the harmonic mean of precision and recall, balancing both metrics.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            averaging_method (Optional[Literal['micro', 'macro', 'weighted', 'none']]):
                Method for averaging the F1 score across classes. Defaults to 'macro'.

        Returns:
            torch.Tensor: Computed F1 score.
        """
        f1_score = F1Score(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            average=averaging_method,
            threshold=self.threshold,
        )
        return f1_score(preds=predictions, target=labels)

    def _calculate_average_precision(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        averaging_method: Optional[
            Literal["micro", "macro", "weighted", "none"]
        ] = "macro",
    ) -> torch.Tensor:
        """
        Calculate the average precision (AP) for the given predictions and labels.

        Average precision (AP) is a threshold-free metric that measures how well the model ranks its predictions by calculating the precision at different thresholds.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            averaging_method (Optional[Literal['micro', 'macro', 'weighted', 'none']]):
                Method for averaging the AP across classes. Defaults to 'macro'.

        Returns:
            torch.Tensor: Computed AP score.
        """
        ap_score = AveragePrecision(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            average=averaging_method,
        )
        return ap_score(preds=predictions, target=labels)

    def _calculate_auroc(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        averaging_method: Optional[
            Literal["micro", "macro", "weighted", "none"]
        ] = "macro",
    ) -> torch.Tensor:
        """
        Calculate the Area Under the Receiver Operating Characteristic curve (AUROC) for the given predictions and labels.

        AUROC is a threshold-free metric that measures how well the model distinguishes between classes by evaluating its ability to rank true positives higher than false positives across all possible classification thresholds.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            averaging_method (Optional[Literal['micro', 'macro', 'weighted', 'none']]):
                Method for averaging AUROC across classes. Defaults to 'macro'.

        Returns:
            torch.Tensor: Computed AUROC score.
        """
        auroc = AUROC(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            average=averaging_method,
        )
        return auroc(preds=predictions, target=labels)

    def _plot_overall_metrics(
        self, predictions: torch.Tensor, labels: torch.Tensor, cmap: str = "plasma"
    ) -> None:
        """
        Plots a bar chart for overall performance metrics (when `per_class_metrics=False`).

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            cmap (str): Name of the colormap to be used (default is 'plasma').
        """
        metrics_df = self.calculate_metrics(
            predictions=predictions, labels=labels, per_class_metrics=False
        )

        # Extract the metric names and their values
        metrics = metrics_df.index  # Metric names
        values = metrics_df["Overall"].values  # Metric values

        # Dynamically get the colormap based on the provided colormap name
        colormap = plt.get_cmap(cmap)

        # Plot the bar chart with a unique color for each bar
        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color=self.colors)

        # Add titles and labels
        plt.title("Overall metric scores", fontsize=16)
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Score", fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right", fontsize=10)

        # Add gridlines and adjust the layout
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        plt.show()

    def _plot_metrics_per_class(
        self, predictions: torch.Tensor, labels: torch.Tensor, cmap: str = "plasma"
    ) -> None:
        """
        Plots metric values per class, with each metric represented by a distinct color and line.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            cmap (str): Name of the colormap to be used (default is 'plasma').
        """
        metrics_df = self.calculate_metrics(
            predictions=predictions, labels=labels, per_class_metrics=True
        )

        # Dynamically get the colormap based on the provided colormap name
        colormap = plt.get_cmap(cmap)

        # Define a list of line styles
        line_styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

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
                linestyle=line_styles[i],
                color=self.colors[i],
            )

        # Add titles and labels
        plt.title("Metric scores per class", fontsize=16)
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Score", fontsize=12)

        # Show legend
        plt.legend(loc="lower right")

        # Add gridlines and adjust layout
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def _plot_metrics_all_thresholds_overall(
            self,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            cmap: str = "plasma",
    ) -> None:
        """
        Plots metrics across different thresholds from 0 to 1 (exclusive) in 0.05 increments.
        Excludes 'auroc' and 'ap' metrics.
        """
        original_threshold = self.threshold
        thresholds = np.arange(0.05, 1.0, 0.05)
        metrics_to_plot = [m for m in self.metrics if m not in ['auroc', 'ap']]

        # Prepare line styles and colors
        line_styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]

        plt.figure(figsize=(10, 6))

        # Initialize a dictionary to store metric values for each metric
        metric_values_dict = {metric_name: [] for metric_name in metrics_to_plot}

        for thresh in thresholds:
            # Temporarily set the threshold
            self.threshold = thresh
            # Calculate all metrics at the current threshold
            metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics=False)
            # Loop through each metric and collect its value
            for metric_name in metrics_to_plot:
                # Map metric names to match DataFrame index
                metric_label = metric_name.capitalize() if metric_name != 'f1' else 'F1'
                value = metrics_df.loc[metric_label, 'Overall']
                metric_values_dict[metric_name].append(value)

        # Reset the threshold to its original value
        self.threshold = original_threshold

        # Plot each metric across thresholds
        for i, metric_name in enumerate(metrics_to_plot):
            metric_values = metric_values_dict[metric_name]
            line_style = line_styles[i % len(line_styles)]
            plt.plot(
                thresholds,
                metric_values,
                label=metric_name.capitalize() if metric_name != 'f1' else 'F1',
                linestyle=line_style,
                linewidth=2,
                color=self.colors[i],
            )

        plt.title("Metrics across different thresholds", fontsize=16)
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Metric Score", fontsize=12)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _plot_metrics_all_thresholds_per_class(
            self,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            cmap: str = "plasma",
    ) -> None:
        """
        Plots metrics across different thresholds from 0 to 1 (exclusive) in 0.05 increments, per class,
        using subplots to display all class plots in a single figure.

        Excludes 'auroc' and 'ap' metrics.
        """
        original_threshold = self.threshold
        thresholds = np.arange(0.05, 1.0, 0.05)
        metrics_to_plot = [m for m in self.metrics if m not in ['auroc', 'ap']]

        class_names = self.classes if self.classes else [f"Class {i}" for i in range(self.num_classes)]
        num_classes = self.num_classes

        # Determine grid size based on the number of classes
        n_cols = int(np.ceil(np.sqrt(num_classes)))
        n_rows = int(np.ceil(num_classes / n_cols))

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

        # If there's only one class, axes is not an array, so we wrap it in a list
        if num_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Prepare line styles and colors
        line_styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (5, 5)), (0, (3, 5, 1, 5))]
        colormap = plt.get_cmap(cmap)

        for class_idx, class_name in enumerate(class_names):
            ax = axes[class_idx]
            # Initialize a dictionary to store metric values for each metric
            metric_values_dict = {metric_name: [] for metric_name in metrics_to_plot}

            for thresh in thresholds:
                # Temporarily set the threshold
                self.threshold = thresh
                # Calculate all metrics at the current threshold
                metrics_df = self.calculate_metrics(predictions, labels, per_class_metrics=True)
                # Loop through each metric and collect its value for the current class
                for metric_name in metrics_to_plot:
                    metric_label = metric_name.capitalize() if metric_name != 'f1' else 'F1'
                    value = metrics_df.loc[metric_label, class_name]
                    metric_values_dict[metric_name].append(value)

            # Reset the threshold to its original value
            self.threshold = original_threshold

            # Plot each metric for the current class
            for i, metric_name in enumerate(metrics_to_plot):
                metric_values = metric_values_dict[metric_name]
                line_style = line_styles[i % len(line_styles)]
                ax.plot(
                    thresholds,
                    metric_values,
                    label=metric_name.capitalize() if metric_name != 'f1' else 'F1',
                    linestyle=line_style,
                    linewidth=2,
                    color=self.colors[i],
                )

            ax.set_title(f"{class_name}", fontsize=12)
            ax.set_xlabel("Threshold", fontsize=10)
            ax.set_ylabel("Metric Score", fontsize=10)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True)

        # Hide any unused subplots
        if num_classes > 1:
            for j in range(num_classes, len(axes)):
                fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_metrics_all_thresholds(
            self,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            per_class_metrics: bool = False,
            cmap: str = "plasma",
    ) -> None:
        """
        Plot performance metrics across thresholds for the given predictions and labels.

        This method provides two modes of plotting:
        - Per-class metrics: Plots the values of metrics for each class individually.
        - Overall metrics: Plots the aggregate metrics over all classes.

        Args:
            predictions (torch.Tensor): Model output predictions of shape (batch_size, num_classes).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
            per_class_metrics (bool): If True, plots metrics for each class individually. If False, plots overall metrics. Defaults to False.
            cmap (str): Name of the colormap to be used (default is 'plasma').

        Returns:
            None
        """
        if per_class_metrics:
            self._plot_metrics_all_thresholds_per_class(predictions=predictions, labels=labels, cmap=cmap)
        else:
            self._plot_metrics_all_thresholds_overall(predictions=predictions, labels=labels, cmap=cmap)

    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        per_class_metrics: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate multiple performance metrics for the given predictions and labels.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
            per_class_metrics (bool): If True, returns metrics for each class individually. If False, returns overall metrics.

        Returns:
            pd.DataFrame: A DataFrame containing the computed metrics based on the metrics tuple (F1, recall, precision, AP, AUROC, accuracy).
        """
        averaging_method = None if per_class_metrics else "macro"
        metrics_results = {}

        if "recall" in self.metrics:
            recall = self._calculate_recall(
                predictions=predictions,
                labels=labels,
                averaging_method=averaging_method,
            )
            metrics_results["Recall"] = recall.cpu().numpy()

        if "precision" in self.metrics:
            precision = self._calculate_precision(
                predictions=predictions,
                labels=labels,
                averaging_method=averaging_method,
            )
            metrics_results["Precision"] = precision.cpu().numpy()

        if "f1" in self.metrics:
            f1_score = self._calculate_f1_score(
                predictions=predictions,
                labels=labels,
                averaging_method=averaging_method,
            )
            metrics_results["F1"] = f1_score.cpu().numpy()

        if "ap" in self.metrics:
            ap_score = self._calculate_average_precision(
                predictions=predictions,
                labels=labels,
                averaging_method=averaging_method,
            )
            metrics_results["AP"] = ap_score.cpu().numpy()

        if "auroc" in self.metrics:
            auroc = self._calculate_auroc(
                predictions=predictions,
                labels=labels,
                averaging_method=averaging_method,
            )
            metrics_results["AUROC"] = auroc.cpu().numpy()

        if "accuracy" in self.metrics:
            accuracy = self._calculate_accuracy(
                predictions=predictions,
                labels=labels,
                averaging_method=averaging_method,
            )
            metrics_results["Accuracy"] = accuracy.cpu().numpy()

        if per_class_metrics:
            if self.classes:
                columns = self.classes
            else:
                columns = None
        else:
            columns = ["Overall"]

        return pd.DataFrame.from_dict(metrics_results, orient="index", columns=columns)

    def plot_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        per_class_metrics: bool = False,
        cmap: str = "plasma"
    ) -> None:
        """
        Plot performance metrics for the given predictions and labels.

        This method provides two modes of plotting:
        - Per-class metrics: Plots the values of metrics for each class individually.
        - Overall metrics: Plots the aggregate metrics over all classes.

        Args:
            predictions (torch.Tensor): Model output predictions of shape (batch_size, num_classes).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_classes).
            per_class_metrics (bool): If True, plots metrics for each class individually. If False, plots overall metrics. Defaults to False.
            cmap (str): Name of the colormap to be used (default is 'plasma').

        Returns:
            None
        """

        # If per-class metrics are requested, plot metric values for each class
        if per_class_metrics:
            self._plot_metrics_per_class(predictions=predictions, labels=labels, cmap=cmap)
        else:
            # Otherwise, plot the overall metrics across all classes
            self._plot_overall_metrics(predictions=predictions, labels=labels, cmap=cmap)

    def plot_confusion_matrix(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Plot confusion matrices for each class in a single figure with multiple subplots.

        Args:
            predictions (torch.Tensor): Model output predictions.
            labels (torch.Tensor): Ground truth labels.
        """
        confusion_matrix = ConfusionMatrix(
            task=self.task,
            num_classes=self.num_classes,
            num_labels=self.num_classes,
            threshold=self.threshold,
            normalize="true",
        )
        confusion_matrix.update(preds=predictions, target=labels)
        conf_mat = confusion_matrix.compute().cpu().numpy()

        if self.task == "binary":
            # For binary classification, conf_mat is a 2x2 matrix
            plt.figure(figsize=(4, 4))
            sns.heatmap(conf_mat, annot=True, fmt=".2f", cmap="Reds", cbar=False)
            plt.title(f"{self.classes[0]}")
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.tight_layout()
            plt.show()
        else:
            # For multilabel classification, conf_mat is (num_labels, 2, 2)
            num_labels = conf_mat.shape[0]
            class_names = self.classes if self.classes else [f"Label {i}" for i in range(num_labels)]

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
                ax.set_xlabel("Predicted class")
                ax.set_ylabel("True class")

            # Hide any unused subplots
            for j in range(num_labels, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()


