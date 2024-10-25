"""
metrics.py

Module containing functions to calculate various performance metrics using torchmetrics.
"""

from typing import Optional, Literal
import torch
import numpy as np
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)


def calculate_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: int,
    threshold: float,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate accuracy for the given predictions and labels.
    """
    accuracy = Accuracy(
        task=task,
        num_classes=num_classes,
        num_labels=num_classes,
        average=averaging_method,
        threshold=threshold,
    )
    return accuracy(preds=predictions, target=labels).cpu().numpy()


def calculate_recall(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: int,
    threshold: float,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate recall for the given predictions and labels.
    """
    recall = Recall(
        task=task,
        num_classes=num_classes,
        num_labels=num_classes,
        average=averaging_method,
        threshold=threshold,
    )
    return recall(preds=predictions, target=labels).cpu().numpy()


def calculate_precision(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: int,
    threshold: float,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate precision for the given predictions and labels.
    """
    precision = Precision(
        task=task,
        num_classes=num_classes,
        num_labels=num_classes,
        average=averaging_method,
        threshold=threshold,
    )
    return precision(preds=predictions, target=labels).cpu().numpy()


def calculate_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: int,
    threshold: float,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate the F1 score for the given predictions and labels.
    """
    f1_score = F1Score(
        task=task,
        num_classes=num_classes,
        num_labels=num_classes,
        average=averaging_method,
        threshold=threshold,
    )
    return f1_score(preds=predictions, target=labels).cpu().numpy()


def calculate_average_precision(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: int,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate the average precision (AP) for the given predictions and labels.
    """
    ap_score = AveragePrecision(
        task=task,
        num_classes=num_classes,
        num_labels=num_classes,
        average=averaging_method,
    )
    return ap_score(preds=predictions, target=labels).cpu().numpy()


def calculate_auroc(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: int,
    averaging_method: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
) -> np.ndarray:
    """
    Calculate the Area Under the Receiver Operating Characteristic curve (AUROC) for the given predictions and labels.
    """
    auroc = AUROC(
        task=task,
        num_classes=num_classes,
        num_labels=num_classes,
        average=averaging_method,
    )
    return auroc(preds=predictions, target=labels).cpu().numpy()
