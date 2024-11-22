# performance_core.py

import argparse
import json
import os

# Import DataProcessor and PerformanceAssessor
from preprocessing.data_processor import DataProcessor
from assessment.performance_assessor import PerformanceAssessor


def process_data(
    annotation_path,
    prediction_path,
    mapping_path=None,
    sample_duration=3.0,
    min_overlap=0.5,
    recording_duration=None,
    columns_annotations=None,
    columns_predictions=None,
    selected_classes=None,
    selected_recordings=None,
    metrics_list=('accuracy', 'precision', 'recall'),
    threshold=0.1,
    class_wise=False
):
    # Load class mapping if provided
    if mapping_path:
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None

    # Check if the paths refer to files or directories
    annotation_dir, annotation_file = (
        (os.path.dirname(annotation_path), os.path.basename(annotation_path))
        if os.path.isfile(annotation_path)
        else (annotation_path, None)
    )

    prediction_dir, prediction_file = (
        (os.path.dirname(prediction_path), os.path.basename(prediction_path))
        if os.path.isfile(prediction_path)
        else (prediction_path, None)
    )

    # Initialize DataProcessor
    processor = DataProcessor(
        prediction_directory_path=prediction_dir,
        prediction_file_name=prediction_file,
        annotation_directory_path=annotation_dir,
        annotation_file_name=annotation_file,
        class_mapping=class_mapping,
        sample_duration=sample_duration,
        min_overlap=min_overlap,
        columns_predictions=columns_predictions,
        columns_annotations=columns_annotations,
        recording_duration=recording_duration,
    )

    # Get available classes and recordings
    available_classes = processor.classes
    available_recordings = processor.samples_df['filename'].unique().tolist()

    # If selected_classes or selected_recordings are None, select all
    if selected_classes is None:
        selected_classes = available_classes
    if selected_recordings is None:
        selected_recordings = available_recordings

    # Get predictions and labels
    predictions, labels, classes = processor.get_filtered_tensors(selected_classes, selected_recordings)

    num_classes = len(classes)
    task = 'binary' if num_classes == 1 else 'multilabel'

    # Initialize PerformanceAssessor
    pa = PerformanceAssessor(
        num_classes=num_classes,
        threshold=threshold,
        classes=classes,
        task=task,
        metrics_list=metrics_list,
    )

    # Calculate metrics
    metrics_df = pa.calculate_metrics(predictions, labels, per_class_metrics=class_wise)

    return metrics_df, pa, predictions, labels


def main():
    # Command-line interface to call process_data
    parser = argparse.ArgumentParser(description='Performance Assessor Core Script')
    # Add arguments
    parser.add_argument('--annotation_path', required=True, help='Path to annotation file or folder')
    parser.add_argument('--prediction_path', required=True, help='Path to prediction file or folder')
    parser.add_argument('--mapping_path', help='Path to class mapping JSON file (optional)')
    parser.add_argument('--sample_duration', type=float, default=3.0, help='Sample duration in seconds')
    parser.add_argument('--min_overlap', type=float, default=0.5, help='Minimum overlap in seconds')
    parser.add_argument('--recording_duration', type=float, help='Recording duration in seconds')
    parser.add_argument('--columns_annotations', type=json.loads, help='JSON string for columns_annotations')
    parser.add_argument('--columns_predictions', type=json.loads, help='JSON string for columns_predictions')
    parser.add_argument('--selected_classes', nargs='+', help='List of selected classes')
    parser.add_argument('--selected_recordings', nargs='+', help='List of selected recordings')
    parser.add_argument('--metrics', nargs='+', default=['accuracy', 'precision', 'recall'], help='List of metrics')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold value (0-1)')
    parser.add_argument('--class_wise', action='store_true', help='Calculate class-wise metrics')
    parser.add_argument('--plot_metrics', action='store_true', help='Plot metrics')
    parser.add_argument('--plot_confusion_matrix', action='store_true', help='Plot confusion matrix')
    parser.add_argument('--plot_metrics_all_thresholds', action='store_true', help='Plot metrics for all thresholds')
    parser.add_argument('--output_dir', help='Directory to save plots')

    args = parser.parse_args()

    metrics_df, pa, predictions, labels = process_data(
        annotation_path=args.annotation_path,
        prediction_path=args.prediction_path,
        mapping_path=args.mapping_path,
        sample_duration=args.sample_duration,
        min_overlap=args.min_overlap,
        recording_duration=args.recording_duration,
        columns_annotations=args.columns_annotations,
        columns_predictions=args.columns_predictions,
        selected_classes=args.selected_classes,
        selected_recordings=args.selected_recordings,
        metrics_list=args.metrics,
        threshold=args.threshold,
        class_wise=args.class_wise
    )

    print(metrics_df)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.plot_metrics:
        pa.plot_metrics(predictions, labels, per_class_metrics=args.class_wise)
        if args.output_dir:
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(args.output_dir, 'metrics_plot.png'))
        else:
            plt.show()

    if args.plot_confusion_matrix:
        pa.plot_confusion_matrix(predictions, labels)
        if args.output_dir:
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
        else:
            plt.show()

    if args.plot_metrics_all_thresholds:
        pa.plot_metrics_all_thresholds(predictions, labels, per_class_metrics=args.class_wise)
        if args.output_dir:
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(args.output_dir, 'metrics_all_thresholds.png'))
        else:
            plt.show()


if __name__ == '__main__':
    main()
