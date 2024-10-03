import os
import re
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch


class DataProcessor:
    """
    Processor for handling and transforming sample data with annotations and predictions.

    This class processes prediction and annotation data, aligning them with sampled time intervals,
    and generates tensors for further model training or evaluation.
    """

    def __init__(
            self,
            prediction_directory_path: str,
            prediction_file_name: Optional[str],
            annotation_directory_path: str,
            annotation_file_name: Optional[str],
            class_mapping: Optional[Dict[str, str]] = None,
            sample_duration: int = 3,
            min_overlap: float = 0.5,
            columns_predictions: Dict[str, str] = None,
            columns_annotations: Dict[str, str] = None,
    ) -> None:
        """
        Initializes the DataProcessor by loading prediction and annotation data.

        Args:
            prediction_directory_path (str): Path to the folder containing prediction files.
            prediction_file_name (Optional[str]): Name of the prediction file to process. If None, all files in the folder are processed.
            annotation_directory_path (str): Path to the folder containing annotation files.
            annotation_file_name (Optional[str]): Name of the annotation file to process. If None, all files in the folder are processed.
            class_mapping (Optional[dict]): Optional dictionary mapping raw class names to standardized class names. Defaults to None.
            sample_duration (int, optional): Length of each data sample in seconds. Defaults to 3.
            min_overlap (float, optional): Minimum overlap required between prediction and annotation to consider a match. Defaults to 0.5.
            columns_predictions (Dict[str, str], optional): Column name mappings for prediction files.
            columns_annotations (Dict[str, str], optional): Column name mappings for annotation files.
        """
        self.sample_duration = sample_duration
        self.min_overlap = min_overlap
        self.class_mapping = class_mapping
        self.columns_predictions = columns_predictions or {}
        self.columns_annotations = columns_annotations or {}

        self.default_columns_predictions = {
            "Start Time": "Begin Time (s)",
            "End Time": "End Time (s)",
            "Class": "Common Name",
            "Recording": "Begin File",
            "Duration": "File Duration (s)",
            "Confidence": "Confidence"
        }

        self.default_columns_annotations = {
            "Start Time": "Begin Time (s)",
            "End Time": "End Time (s)",
            "Class": "Class",
            "Recording": "Begin File",
            "Duration": "File Duration (s)"
        }

        if prediction_file_name is None or annotation_file_name is None:
            # Read all prediction files into a single DataFrame
            predictions_df = self._read_and_concatenate_files_in_directory(
                prediction_directory_path
            )
            # Read all annotation files into a single DataFrame
            annotations_df = self._read_and_concatenate_files_in_directory(
                annotation_directory_path
            )

            # For predictions
            recording_col_pred = self.get_column_name("Recording", prediction=True)
            if recording_col_pred in predictions_df.columns:
                predictions_df['recording_filename'] = self._extract_recording_filename(
                    predictions_df[recording_col_pred])
            elif 'Begin Path' in predictions_df.columns:
                predictions_df['recording_filename'] = self._extract_recording_filename(
                    predictions_df['Begin Path'])
            else:
                predictions_df['recording_filename'] = self._extract_recording_filename_from_filename(
                    predictions_df['source_file'])

            # For annotations
            recording_col_annot = self.get_column_name("Recording", prediction=False)
            if recording_col_annot in annotations_df.columns:
                annotations_df['recording_filename'] = self._extract_recording_filename(
                    annotations_df[recording_col_annot])
            elif 'Begin Path' in annotations_df.columns:
                annotations_df['recording_filename'] = self._extract_recording_filename(
                    annotations_df['Begin Path'])
            else:
                annotations_df['recording_filename'] = self._extract_recording_filename_from_filename(
                    annotations_df['source_file'])

            # Apply class mapping to prediction files only
            if self.class_mapping:
                class_col_pred = self.get_column_name("Class", prediction=True)
                predictions_df[class_col_pred] = predictions_df[
                    class_col_pred
                ].apply(lambda x: self.class_mapping.get(x, x))

            # Collect all classes from both prediction and annotation
            class_col_pred = self.get_column_name("Class", prediction=True)
            class_col_annot = self.get_column_name("Class", prediction=False)

            pred_classes = set(predictions_df[class_col_pred].unique())
            annot_classes = set(annotations_df[class_col_annot].unique())
            all_classes = pred_classes.union(annot_classes)
            self.classes = tuple(sorted(all_classes))

            self.samples_df = pd.DataFrame()

            # Get the set of all recording filenames
            recording_filenames = set(predictions_df['recording_filename'].unique()).union(
                set(annotations_df['recording_filename'].unique())
            )

            # Process each recording
            for recording_filename in recording_filenames:
                pred_df = predictions_df[predictions_df['recording_filename'] == recording_filename]
                annot_df = annotations_df[annotations_df['recording_filename'] == recording_filename]

                # Determine file duration
                file_duration_col_pred = self.get_column_name("Duration", prediction=True)
                end_time_col_pred = self.get_column_name("End Time", prediction=True)
                end_time_col_annot = self.get_column_name("End Time", prediction=False)

                if file_duration_col_pred in pred_df.columns and not pred_df[file_duration_col_pred].isnull().all():
                    file_duration = pred_df[file_duration_col_pred].iloc[0]
                else:
                    pred_max_time = pred_df[end_time_col_pred].max() if not pred_df.empty else 0
                    annot_max_time = annot_df[end_time_col_annot].max() if not annot_df.empty else 0
                    file_duration = max(pred_max_time, annot_max_time)

                # Initialize the DataFrame for sampled intervals, including all classes
                samples_df = self._initialize_samples(
                    filename=recording_filename, file_duration=file_duration
                )

                # Update samples_df with predictions and annotations
                self._update_samples_with_predictions(pred_df, samples_df)
                self._update_samples_with_annotations(annot_df, samples_df)

                # Append samples_df to self.samples_df
                self.samples_df = pd.concat([self.samples_df, samples_df], ignore_index=True)

            # After processing all recordings, create tensors
            self.prediction_tensors = torch.tensor(
                self.samples_df[
                    [f"{label}_confidence" for label in self.classes]
                ].values,
                dtype=torch.float32,
            )

            self.label_tensors = torch.tensor(
                self.samples_df[
                    [f"{label}_annotation" for label in self.classes]
                ].values,
                dtype=torch.int64,
            )

        else:
            # Ensure that the prediction and annotation files match if specified
            if not prediction_file_name.startswith(
                    annotation_file_name.split(".")[0]
            ):
                warnings.warn(
                    "Prediction file name and annotation file name do not fully match, but proceeding anyway."
                )

            # Process individual files
            prediction_file = os.path.join(
                prediction_directory_path, prediction_file_name
            )
            annotation_file = os.path.join(
                annotation_directory_path, annotation_file_name
            )
            pred_df = pd.read_csv(prediction_file, sep="\t")
            annot_df = pd.read_csv(annotation_file, sep="\t")

            # Add source_file column
            pred_df['source_file'] = prediction_file_name
            annot_df['source_file'] = annotation_file_name

            # Extract recording filenames
            recording_col_pred = self.get_column_name("Recording", prediction=True)
            if recording_col_pred in pred_df.columns:
                pred_df['recording_filename'] = self._extract_recording_filename(pred_df[recording_col_pred])
            elif 'Begin Path' in pred_df.columns:
                pred_df['recording_filename'] = self._extract_recording_filename(pred_df['Begin Path'])
            else:
                pred_df['recording_filename'] = self._extract_recording_filename_from_filename(pred_df['source_file'])

            recording_col_annot = self.get_column_name("Recording", prediction=False)
            if recording_col_annot in annot_df.columns:
                annot_df['recording_filename'] = self._extract_recording_filename(annot_df[recording_col_annot])
            elif 'Begin Path' in annot_df.columns:
                annot_df['recording_filename'] = self._extract_recording_filename(annot_df['Begin Path'])
            else:
                annot_df['recording_filename'] = self._extract_recording_filename_from_filename(annot_df['source_file'])

            # Apply class mapping to prediction files only
            if self.class_mapping:
                class_col_pred = self.get_column_name("Class", prediction=True)
                pred_df[class_col_pred] = pred_df[
                    class_col_pred
                ].apply(lambda x: self.class_mapping.get(x, x))

            # Collect all classes from both prediction and annotation
            class_col_pred = self.get_column_name("Class", prediction=True)
            class_col_annot = self.get_column_name("Class", prediction=False)

            pred_classes = set(pred_df[class_col_pred].unique())
            annot_classes = set(annot_df[class_col_annot].unique())
            all_classes = pred_classes.union(annot_classes)
            self.classes = tuple(sorted(all_classes))

            self.samples_df = pd.DataFrame()

            recording_filenames = set(pred_df['recording_filename'].unique()).union(
                set(annot_df['recording_filename'].unique())
            )

            # Process each recording
            for recording_filename in recording_filenames:
                pred_sub_df = pred_df[pred_df['recording_filename'] == recording_filename]
                annot_sub_df = annot_df[annot_df['recording_filename'] == recording_filename]

                # Determine file duration
                file_duration_col_pred = self.get_column_name("Duration", prediction=True)
                end_time_col_pred = self.get_column_name("End Time", prediction=True)
                end_time_col_annot = self.get_column_name("End Time", prediction=False)

                if file_duration_col_pred in pred_sub_df.columns and not pred_sub_df[file_duration_col_pred].isnull().all():
                    file_duration = pred_sub_df[file_duration_col_pred].iloc[0]
                else:
                    pred_max_time = pred_sub_df[end_time_col_pred].max() if not pred_sub_df.empty else 0
                    annot_max_time = annot_sub_df[end_time_col_annot].max() if not annot_sub_df.empty else 0
                    file_duration = max(pred_max_time, annot_max_time)

                # Initialize the DataFrame for sampled intervals, including all classes
                samples_df = self._initialize_samples(
                    filename=recording_filename, file_duration=file_duration
                )

                # Update samples_df with predictions and annotations
                self._update_samples_with_predictions(pred_sub_df, samples_df)
                self._update_samples_with_annotations(annot_sub_df, samples_df)

                # Append samples_df to self.samples_df
                self.samples_df = pd.concat([self.samples_df, samples_df], ignore_index=True)

            # Create tensors
            self.prediction_tensors = torch.tensor(
                self.samples_df[
                    [f"{label}_confidence" for label in self.classes]
                ].values,
                dtype=torch.float32,
            )

            self.label_tensors = torch.tensor(
                self.samples_df[
                    [f"{label}_annotation" for label in self.classes]
                ].values,
                dtype=torch.int64,
            )

    def _extract_recording_filename(self, path_column: pd.Series) -> pd.Series:
        """
        Extracts the recording filename from a path column.

        Args:
            path_column (pd.Series): Series containing file paths.

        Returns:
            pd.Series: Series containing extracted recording filenames.
        """
        return path_column.apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    def _extract_recording_filename_from_filename(self, filename_series: pd.Series) -> pd.Series:
        """
        Extracts the recording filename from the source file name.

        Args:
            filename_series (pd.Series): Series containing file names.

        Returns:
            pd.Series: Series containing extracted recording filenames.
        """
        return filename_series.apply(lambda x: x.split('.')[0])

    def _read_and_concatenate_files_in_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Reads all .txt files from a directory and concatenates them into a single DataFrame.

        Args:
            directory_path (str): Path to the directory containing the .txt files.

        Returns:
            pd.DataFrame: A concatenated DataFrame of all files.
        """
        df_list = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory_path, filename)
                df = pd.read_csv(filepath, sep="\t")
                df['source_file'] = filename  # Add the filename as a column
                df_list.append(df)
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()

    def _initialize_samples(self, filename: str, file_duration: float) -> pd.DataFrame:
        """
        Initializes a DataFrame of time-based sample intervals using the file duration and sample duration.

        Args:
            filename (str): The name of the file for which samples are being created.
            file_duration (float): The total duration of the file in seconds.

        Returns:
            pd.DataFrame: DataFrame containing initialized sample intervals, confidence scores, and annotations.
        """
        intervals = np.arange(0, file_duration, self.sample_duration)
        samples = {
            "filename": filename,
            "sample_index": [],
            "start_time": [],
            "end_time": [],
        }

        for idx, start in enumerate(intervals):
            samples["sample_index"].append(idx)
            samples["start_time"].append(start)
            samples["end_time"].append(
                min(start + self.sample_duration, file_duration)
            )

        # Initialize columns for confidence scores and annotations for each class
        for label in self.classes:
            samples[f"{label}_confidence"] = [0.0] * len(intervals)  # Float values
            samples[f"{label}_annotation"] = [0] * len(intervals)    # Integer values

        return pd.DataFrame(samples)

    def _update_samples_with_predictions(
            self, pred_df: pd.DataFrame, samples_df: pd.DataFrame
    ) -> None:
        """
        Updates the samples_df with prediction confidence scores from pred_df.
        """

        class_col = self.get_column_name("Class", prediction=True)
        start_time_col = self.get_column_name("Start Time", prediction=True)
        end_time_col = self.get_column_name("End Time", prediction=True)
        confidence_col = self.get_column_name("Confidence", prediction=True)

        for _, row in pred_df.iterrows():
            class_name = row[class_col]
            begin_time = row[start_time_col]
            end_time = row[end_time_col]
            confidence = row[confidence_col]

            sample_indices = samples_df[
                (samples_df["start_time"] <= begin_time)
                & (samples_df["end_time"] >= end_time)
                ].index
            for i in sample_indices:
                samples_df.loc[i, f"{class_name}_confidence"] = confidence

    def _update_samples_with_annotations(
            self, annot_df: pd.DataFrame, samples_df: pd.DataFrame
    ) -> None:
        """
        Updates the samples_df with annotations from annot_df.
        """

        class_col = self.get_column_name("Class", prediction=False)
        start_time_col = self.get_column_name("Start Time", prediction=False)
        end_time_col = self.get_column_name("End Time", prediction=False)

        for _, row in annot_df.iterrows():
            class_name = row[class_col]
            begin_time = row[start_time_col]
            end_time = row[end_time_col]

            sample_indices = samples_df[
                (
                    samples_df["start_time"] <= end_time - self.min_overlap
                ) & (
                    samples_df["end_time"] >= begin_time + self.min_overlap
                )
                ].index
            for i in sample_indices:
                samples_df.loc[i, f"{class_name}_annotation"] = 1

    def get_column_name(self, field_name: str, prediction=True) -> str:
        """
        Retrieves the column name for a given field, considering user-specified and default mappings.

        Args:
            field_name (str): The logical name of the field (e.g., "Class", "Start Time").
            prediction (bool, optional): Whether to retrieve from predictions or annotations. Defaults to True.

        Returns:
            str: The actual column name in the DataFrame.
        """
        if prediction:
            return self.columns_predictions.get(
                field_name,
                self.default_columns_predictions.get(field_name, field_name)
            )
        else:
            return self.columns_annotations.get(
                field_name,
                self.default_columns_annotations.get(field_name, field_name)
            )

    def get_sample_data(self) -> pd.DataFrame:
        """
        Retrieves the DataFrame containing all the sample intervals, prediction scores, and annotations.

        Returns:
            pd.DataFrame: The DataFrame representing sampled data.
        """
        return self.samples_df

    def get_prediction_tensor(self) -> torch.Tensor:
        """
        Returns the tensor containing prediction confidence scores for each class and sample.

        Returns:
            torch.Tensor: Tensor containing the prediction confidence scores.
        """
        return self.prediction_tensors

    def get_label_tensor(self) -> torch.Tensor:
        """
        Returns the tensor containing binary labels for each class and sample.

        Returns:
            torch.Tensor: Tensor containing the label data.
        """
        return self.label_tensors
