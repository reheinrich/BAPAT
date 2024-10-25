import pytest
import torch
import numpy as np
from bapat.assessment.metrics import (
    calculate_accuracy,
    calculate_recall,
    calculate_precision,
    calculate_f1_score,
    calculate_average_precision,
    calculate_auroc,
)


class TestCalculateAccuracy:
    def test_binary_classification_perfect(self):
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = torch.tensor([0.6, 0.4, 0.3, 0.7])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
            averaging_method="micro",
        )
        # Expected accuracy: 2 correct out of 4
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        labels = torch.tensor([0, 0, 0, 0])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_all_ones(self):
        predictions = torch.tensor([0.6, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_accuracy(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_perfect(self):
        predictions = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_accuracy(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_imperfect(self):
        predictions = torch.tensor([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_accuracy(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
            averaging_method="micro",
        )
        # Expected accuracy: 1 correct out of 3
        assert np.isclose(result, 0.3333333333)

    def test_multilabel_classification_perfect(self):
        predictions = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        labels = torch.tensor([[1, 0], [0, 1], [1, 0]])
        result = calculate_accuracy(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
            averaging_method="micro",
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = torch.tensor([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        labels = torch.tensor([[1, 0], [0, 1], [1, 0]])
        result = calculate_accuracy(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
            averaging_method="micro",
        )
        # Expected accuracy: 5 correct out of 6 = 0.8333333
        assert np.isclose(result, 0.8333333)

    def test_incorrect_shapes(self):
        predictions = torch.tensor([0.9, 0.2, 0.8])
        labels = torch.tensor([1, 0])
        with pytest.raises(RuntimeError, match="same shape"):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        with pytest.raises(
            ValueError,
            match="Expected argument `threshold` to be a float in the \\[0,1\\] range",
        ):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=1.5,
            )

    def test_non_tensor_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        with pytest.raises(AttributeError):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_empty_tensors(self):
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with pytest.raises(RuntimeError):
            calculate_accuracy(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )


class TestCalculateRecall:
    def test_binary_classification_perfect(self):
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = torch.tensor([0.6, 0.4, 0.3, 0.7])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        # True Positives: 1, False Negatives: 1, Recall = 1 / (1 + 1) = 0.5
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        labels = torch.tensor([0, 0, 0, 0])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        # No true positives, recall is undefined but torchmetrics returns 0
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = torch.tensor([0.6, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_recall(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_perfect(self):
        predictions = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_recall(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_imperfect(self):
        predictions = torch.tensor([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_recall(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
        )
        # Recall depends on per-class performance; compute manually if needed
        assert np.isclose(result, 0.3333333333)

    def test_multilabel_classification_perfect(self):
        predictions = torch.tensor([[0.9, 0.8], [0.8, 0.9], [0.7, 0.6]])
        labels = torch.tensor([[1, 1], [1, 1], [1, 1]])
        result = calculate_recall(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = torch.tensor([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        labels = torch.tensor([[1, 0], [0, 1], [1, 0]])
        result = calculate_recall(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
        )
        # Expected recall: (1.0 + 0.5) / 2 = 0.75
        assert np.isclose(result, 0.75)

    def test_incorrect_shapes(self):
        predictions = torch.tensor([0.9, 0.2, 0.8])
        labels = torch.tensor([1, 0])
        with pytest.raises(RuntimeError, match="same shape"):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        with pytest.raises(
            ValueError,
            match="Expected argument `threshold` to be a float in the \\[0,1\\] range",
        ):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=-0.1,
            )

    def test_non_tensor_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        with pytest.raises(AttributeError):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_empty_tensors(self):
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with pytest.raises(RuntimeError):
            calculate_recall(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )


class TestCalculatePrecision:
    def test_binary_classification_perfect(self):
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = torch.tensor([0.6, 0.4, 0.3, 0.7])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        # True Positives: 1, False Positives: 1, Precision = 1 / (1 + 1) = 0.5
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        labels = torch.tensor([0, 0, 0, 0])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        # No predicted positives, precision is undefined but torchmetrics returns 0
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = torch.tensor([0.6, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_perfect(self):
        predictions = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_precision(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_imperfect(self):
        predictions = torch.tensor([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_precision(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
        )
        # Precision depends on per-class performance; compute manually if needed
        assert np.isclose(result, 0.3333333333)

    def test_multilabel_classification_perfect(self):
        predictions = torch.tensor([[0.9, 0.8], [0.8, 0.9], [0.7, 0.6]])
        labels = torch.tensor([[1, 1], [1, 1], [1, 1]])
        result = calculate_precision(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        # Imperfect predictions for multilabel classification
        predictions = torch.tensor(
            [
                [0.9, 0.2],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.5, 0.5],  # This will be thresholded to [0, 0]
            ]
        )
        labels = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [1, 0],
                [0, 1],
            ]
        )
        result = calculate_precision(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
        )
        # Adjusted calculation considering torchmetrics thresholding
        # Class 0: TP=2, FP=0 => Precision=2/2=1.0
        # Class 1: TP=1, FP=1 => Precision=1/2=0.5
        # Macro average precision: (1.0 + 0.5)/2 = 0.75
        expected_precision = (1.0 + 0.5) / 2
        assert np.isclose(result, expected_precision, atol=1e-4)

    def test_incorrect_shapes(self):
        predictions = torch.tensor([0.9, 0.2, 0.8])
        labels = torch.tensor([1, 0])
        with pytest.raises(RuntimeError, match="same shape"):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        with pytest.raises(
            ValueError,
            match="Expected argument `threshold` to be a float in the \\[0,1\\] range",
        ):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=-0.1,
            )

    def test_non_tensor_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        with pytest.raises(AttributeError):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_empty_tensors(self):
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with pytest.raises(RuntimeError):
            calculate_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )


class TestCalculateF1Score:
    def test_binary_classification_perfect(self):
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = torch.tensor([0.6, 0.4, 0.3, 0.7])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        # Precision and Recall are both 0.5, so F1 = 0.5
        assert np.isclose(result, 0.5)

    def test_binary_classification_all_zeros(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        labels = torch.tensor([0, 0, 0, 0])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        # No predicted positives and no actual positives, F1 is undefined, torchmetrics returns 0
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = torch.tensor([0.6, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_f1_score(
            predictions,
            labels,
            task="binary",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_perfect(self):
        predictions = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_imperfect(self):
        predictions = torch.tensor([[0.3, 0.5, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        labels = torch.tensor([1, 0, 2])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
            threshold=None,
        )
        # F1 depends on per-class performance; compute manually if needed
        assert np.isclose(result, 0.3333333333)

    def test_multilabel_classification_perfect(self):
        predictions = torch.tensor([[0.9, 0.8], [0.8, 0.9], [0.7, 0.6]])
        labels = torch.tensor([[1, 1], [1, 1], [1, 1]])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        predictions = torch.tensor([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
        labels = torch.tensor([[1, 0], [0, 1], [1, 0]])
        result = calculate_f1_score(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
            threshold=0.5,
        )
        # Expected F1 score: (0.8 + 0.8) / 2 = 0.8333333
        assert np.isclose(result, 0.8333333)

    def test_incorrect_shapes(self):
        predictions = torch.tensor([0.9, 0.2, 0.8])
        labels = torch.tensor([1, 0])
        with pytest.raises(RuntimeError, match="same shape"):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_invalid_threshold(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        with pytest.raises(
            ValueError,
            match="Expected argument `threshold` to be a float in the \\[0,1\\] range",
        ):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=1.5,
            )

    def test_non_tensor_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        with pytest.raises(AttributeError):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )

    def test_empty_tensors(self):
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with pytest.raises(RuntimeError):
            calculate_f1_score(
                predictions,
                labels,
                task="binary",
                num_classes=2,
                threshold=0.5,
            )


class TestCalculateAveragePrecision:
    def test_binary_classification_perfect(self):
        predictions = torch.tensor([0.9, 0.8, 0.7, 0.6])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = torch.tensor([0.9, 0.6, 0.3, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        # Average precision is between 0 and 1, less than perfect
        assert np.isclose(result, 0.8333333333)

    def test_binary_classification_all_zeros(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        labels = torch.tensor([0, 0, 0, 0])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        # Should return 0.0 when there are no positive labels
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = torch.tensor([0.6, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_average_precision(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_perfect(self):
        predictions = torch.tensor(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        )
        labels = torch.tensor([0, 1, 2])
        result = calculate_average_precision(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_imperfect(self):
        # Adjusted predictions for an imperfect scenario
        predictions = torch.tensor(
            [
                [0.33, 0.33, 0.34],
                [0.34, 0.33, 0.33],
                [0.33, 0.34, 0.33],
                [0.33, 0.33, 0.34],
                [0.34, 0.33, 0.33],
                [0.33, 0.34, 0.33],
            ]
        )
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        result = calculate_average_precision(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
        )
        # Now the average precision should be less than 1.0
        assert np.isclose(result, 0.3333333)

    def test_multilabel_classification_perfect(self):
        predictions = torch.tensor([[0.9, 0.9], [0.8, 0.8], [0.7, 0.7]])
        labels = torch.tensor([[1, 1], [1, 1], [1, 1]])
        result = calculate_average_precision(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
        )
        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        # Imperfect predictions for multilabel classification
        predictions = torch.tensor(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        labels = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 1],
                [1, 0],
            ]
        )
        result = calculate_average_precision(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
        )

        assert np.isclose(result, 0.8055556)

    def test_incorrect_shapes(self):
        predictions = torch.tensor([0.9, 0.2, 0.8])
        labels = torch.tensor([1, 0])
        with pytest.raises(RuntimeError, match="same shape"):
            calculate_average_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
            )

    def test_non_tensor_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        with pytest.raises(AttributeError):
            calculate_average_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
            )

    def test_empty_tensors(self):
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with pytest.raises(ValueError, match="Expected argument `target` to be an int"):
            calculate_average_precision(
                predictions,
                labels,
                task="binary",
                num_classes=2,
            )


class TestCalculateAUROC:
    def test_binary_classification_perfect(self):
        predictions = torch.tensor([0.9, 0.8, 0.7, 0.6])
        labels = torch.tensor([1, 1, 0, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        assert np.isclose(result, 1.0)

    def test_binary_classification_imperfect(self):
        predictions = torch.tensor([0.9, 0.6, 0.3, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        assert np.isclose(result, 0.75)

    def test_binary_classification_all_zeros(self):
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4])
        labels = torch.tensor([0, 0, 0, 0])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        # torchmetrics returns 0.0 when there are no positive cases
        assert np.isclose(result, 0.0)

    def test_binary_classification_all_ones(self):
        predictions = torch.tensor([0.6, 0.7, 0.8, 0.9])
        labels = torch.tensor([1, 1, 1, 1])
        result = calculate_auroc(
            predictions,
            labels,
            task="binary",
            num_classes=2,
        )
        # torchmetrics returns 0.0 when there are no negative cases
        assert np.isclose(result, 0.0)

    def test_multiclass_classification_perfect(self):
        predictions = torch.tensor(
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        )
        labels = torch.tensor([0, 1, 2])
        result = calculate_auroc(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
        )
        assert np.isclose(result, 1.0)

    def test_multiclass_classification_imperfect(self):
        # Imperfect predictions for multiclass classification
        predictions = torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.4, 0.4, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.3, 0.5],
            ]
        )
        labels = torch.tensor([0, 0, 2, 0, 1, 2])
        result = calculate_auroc(
            predictions,
            labels,
            task="multiclass",
            num_classes=3,
        )

        assert np.isclose(result, 0.8444444)

    def test_multilabel_classification_perfect(self):
        # Perfect predictions for multilabel classification with both positive and negative labels
        predictions = torch.tensor(
            [
                [0.99, 0.99],  # labels [1,1]
                [0.99, 0.01],  # labels [1,0]
                [0.01, 0.99],  # labels [0,1]
                [0.99, 0.99],  # labels [1,1]
                [0.01, 0.01],  # labels [0,0]
            ]
        )
        labels = torch.tensor(
            [
                [1, 1],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0],
            ]
        )
        result = calculate_auroc(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
        )

        assert np.isclose(result, 1.0)

    def test_multilabel_classification_imperfect(self):
        # Imperfect predictions for multilabel classification
        predictions = torch.tensor(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        labels = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 1],
                [1, 0],
            ]
        )
        result = calculate_auroc(
            predictions,
            labels,
            task="multilabel",
            num_classes=2,
        )

        assert np.isclose(result, 0.666666)

    def test_incorrect_shapes(self):
        predictions = torch.tensor([0.9, 0.2])
        labels = torch.tensor([1])
        with pytest.raises(
            IndexError,
            match="index \d+ is out of bounds for dimension \d+ with size \d+",
        ):
            calculate_auroc(
                predictions,
                labels,
                task="binary",
                num_classes=2,
            )

    def test_non_tensor_inputs(self):
        predictions = [0.9, 0.2, 0.8, 0.1]
        labels = [1, 0, 1, 0]
        with pytest.raises(AttributeError):
            calculate_auroc(
                predictions,
                labels,
                task="binary",
                num_classes=2,
            )

    def test_empty_tensors(self):
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with pytest.raises(IndexError, match="index is out of bounds"):
            calculate_auroc(
                predictions,
                labels,
                task="binary",
                num_classes=2,
            )
