# metrics.py

"""
Module containing functions to calculate various performance metrics using scikit-learn.
"""

from typing import Optional, Literal
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


def calculate_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    num_classes: int,
    threshold: float,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate accuracy for the given predictions and labels.
    """
    # Input validation
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    if task == "binary":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        acc = accuracy_score(y_true, y_pred)
        acc = np.array([acc])  # Return as np.ndarray
    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        if averaging_method == "micro":
            correct = (y_pred == y_true).sum()
            total = y_true.size
            acc = correct / total if total > 0 else np.nan
            acc = np.array([acc])
        elif averaging_method == "macro":
            accuracies = []
            for i in range(num_classes):
                acc_i = accuracy_score(y_true[:, i], y_pred[:, i])
                accuracies.append(acc_i)
            acc = np.mean(accuracies)
            acc = np.array([acc])
        elif averaging_method == "weighted":
            accuracies = []
            weights = []
            for i in range(num_classes):
                acc_i = accuracy_score(y_true[:, i], y_pred[:, i])
                accuracies.append(acc_i)
                weights.append(np.sum(y_true[:, i]))
            if sum(weights) == 0:
                acc = np.array([0.0])
            else:
                acc = np.average(accuracies, weights=weights)
                acc = np.array([acc])
        elif averaging_method in [None, "none"]:
            accuracies = []
            for i in range(num_classes):
                acc_i = accuracy_score(y_true[:, i], y_pred[:, i])
                accuracies.append(acc_i)
            acc = np.array(accuracies)
        else:
            raise ValueError(f"Invalid averaging method: {averaging_method}")
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return acc


def calculate_recall(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    threshold: float,
    averaging_method: Optional[
        Literal["binary", "micro", "macro", "weighted", "samples", "none"]
    ] = None,
) -> np.ndarray:
    """
    Calculate recall for the given predictions and labels.
    """
    # Input validation
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    if averaging_method == "none":
        averaging = None
    else:
        averaging = averaging_method

    if task == "binary":
        if averaging is None:
            averaging = "binary"  # Use 'binary' averaging for binary classification
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        recall = recall_score(y_true, y_pred, average=averaging, zero_division=0)
    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        recall = recall_score(y_true, y_pred, average=averaging, zero_division=0)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    if isinstance(recall, np.ndarray):
        return recall

    return np.array([recall])


def calculate_precision(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    threshold: float,
    averaging_method: Optional[
        Literal["binary", "micro", "macro", "weighted", "samples", "none"]
    ] = None,
) -> np.ndarray:
    """
    Calculate precision for the given predictions and labels.
    """
    # Input validation
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    if averaging_method == "none":
        averaging = None
    else:
        averaging = averaging_method

    if task == "binary":
        if averaging is None:
            averaging = "binary"
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        precision = precision_score(y_true, y_pred, average=averaging, zero_division=0)
    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        precision = precision_score(
            y_true, y_pred, average=averaging, zero_division=0
        )
    else:
        raise ValueError(f"Unsupported task type: {task}")

    if isinstance(precision, np.ndarray):
        return precision

    return np.array([precision])


def calculate_f1_score(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    threshold: float,
    averaging_method: Optional[
        Literal["binary", "micro", "macro", "weighted", "samples", "none"]
    ] = None,
) -> np.ndarray:
    """
    Calculate the F1 score for the given predictions and labels.
    """
    # Input validation
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    if averaging_method == "none":
        averaging = None
    else:
        averaging = averaging_method

    if task == "binary":
        if averaging is None:
            averaging = "binary"
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        f1 = f1_score(y_true, y_pred, average=averaging, zero_division=0)
    elif task == "multilabel":
        y_pred = (predictions >= threshold).astype(int)
        y_true = labels.astype(int)
        f1 = f1_score(y_true, y_pred, average=averaging, zero_division=0)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    if isinstance(f1, np.ndarray):
        return f1

    return np.array([f1])


def calculate_average_precision(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    averaging_method: Optional[
        Literal["micro", "macro", "weighted", "samples", "none"]
    ] = None,
) -> np.ndarray:
    """
    Calculate the average precision (AP) for the given predictions and labels.
    """
    # Input validation
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    averaging = None if averaging_method == "none" else averaging_method

    if task == "binary":
        y_true = labels.astype(int)
        y_scores = predictions
        ap = average_precision_score(y_true, y_scores, average=averaging)
    elif task == "multilabel":
        y_true = labels.astype(int)
        y_scores = predictions
        ap = average_precision_score(y_true, y_scores, average=averaging)
    else:
        raise ValueError(f"Unsupported task type for average precision: {task}")

    if isinstance(ap, np.ndarray):
        return ap

    return np.array([ap])


def calculate_auroc(
    predictions: np.ndarray,
    labels: np.ndarray,
    task: Literal["binary", "multilabel"],
    averaging_method: Optional[
        Literal["macro", "weighted", "samples", "none"]
    ] = "macro",
) -> np.ndarray:
    """
    Calculate the Area Under the Receiver Operating Characteristic curve (AUROC) for the given predictions and labels.
    """
    # Input validation
    if predictions.size == 0 or labels.size == 0:
        raise ValueError("Predictions and labels must not be empty.")
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Initialize averaging based on the averaging_method
    averaging = None if averaging_method == "none" else averaging_method

    try:
        if task == "binary":
            y_true = labels.astype(int)
            y_scores = predictions
            auroc = roc_auc_score(y_true, y_scores)
        elif task == "multilabel":
            y_true = labels.astype(int)
            y_scores = predictions
            auroc = roc_auc_score(y_true, y_scores, average=averaging)
        else:
            raise ValueError(f"Unsupported task type: {task}")
    except ValueError as e:
        # Catch specific error related to one class present in y_true
        if "Only one class present in y_true" in str(e):
            # Return NaN for AUROC when only one class is present
            auroc = np.nan
        elif "Number of classes in y_true" in str(e):
            auroc = np.nan
        else:
            # Re-raise other unexpected exceptions
            raise

    if isinstance(auroc, np.ndarray):
        return auroc

    return np.array([auroc])
