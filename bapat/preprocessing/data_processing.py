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
            column_name_predictions: str = "Common Name",
            column_name_annotations: str = "Class",
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
        """
        self.sample_duration = sample_duration
        self.min_overlap = min_overlap
        self.class_mapping = class_mapping
        self.column_name_predictions = column_name_predictions
        self.column_name_annotations = column_name_annotations

        if prediction_file_name is None or annotation_file_name is None:
            # Read all prediction files into a single DataFrame
            predictions_df = self._read_and_concatenate_files_in_directory(
                prediction_directory_path
            )
            # Read all annotation files into a single DataFrame
            annotations_df = self._read_and_concatenate_files_in_directory(
                annotation_directory_path
            )

            # Extract recording filenames from 'Begin Path' in predictions
            if 'Begin Path' in predictions_df.columns:
                predictions_df['recording_filename'] = self._extract_recording_filename(predictions_df['Begin Path'])
            else:
                # Extract from source_file
                predictions_df['recording_filename'] = self._extract_recording_filename_from_filename(predictions_df['source_file'])

            # Extract recording filenames from 'Begin Path' or 'Begin File' in annotations
            if 'Begin Path' in annotations_df.columns:
                annotations_df['recording_filename'] = self._extract_recording_filename(annotations_df['Begin Path'])
            elif 'Begin File' in annotations_df.columns:
                annotations_df['recording_filename'] = self._extract_recording_filename(annotations_df['Begin File'])
            else:
                # Extract from source_file
                annotations_df['recording_filename'] = self._extract_recording_filename_from_filename(annotations_df['source_file'])

            # Apply class mapping to prediction files only
            if self.class_mapping:
                predictions_df[self.column_name_predictions] = predictions_df[
                    self.column_name_predictions
                ].apply(lambda x: self.class_mapping.get(x, x))

            # Collect all classes from both prediction and annotation
            pred_classes = set(predictions_df[self.column_name_predictions].unique())
            annot_classes = set(annotations_df[self.column_name_annotations].unique())
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
                file_duration = None
                if 'File Duration (s)' in pred_df.columns and not pred_df['File Duration (s)'].isnull().all():
                    file_duration = pred_df['File Duration (s)'].iloc[0]
                else:
                    pred_max_time = pred_df['End Time (s)'].max() if not pred_df.empty else 0
                    annot_max_time = annot_df['End Time (s)'].max() if not annot_df.empty else 0
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
            if 'Begin Path' in pred_df.columns:
                pred_df['recording_filename'] = self._extract_recording_filename(pred_df['Begin Path'])
            else:
                pred_df['recording_filename'] = self._extract_recording_filename_from_filename(pred_df['source_file'])

            if 'Begin Path' in annot_df.columns:
                annot_df['recording_filename'] = self._extract_recording_filename(annot_df['Begin Path'])
            elif 'Begin File' in annot_df.columns:
                annot_df['recording_filename'] = self._extract_recording_filename(annot_df['Begin File'])
            else:
                annot_df['recording_filename'] = self._extract_recording_filename_from_filename(annot_df['source_file'])

            # Apply class mapping to prediction files only
            if self.class_mapping:
                pred_df[self.column_name_predictions] = pred_df[
                    self.column_name_predictions
                ].apply(lambda x: self.class_mapping.get(x, x))

            # Collect all classes from both prediction and annotation
            pred_classes = set(pred_df[self.column_name_predictions].unique())
            annot_classes = set(annot_df[self.column_name_annotations].unique())
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
                file_duration = None
                if 'File Duration (s)' in pred_sub_df.columns and not pred_sub_df['File Duration (s)'].isnull().all():
                    file_duration = pred_sub_df['File Duration (s)'].iloc[0]
                else:
                    pred_max_time = pred_sub_df['End Time (s)'].max() if not pred_sub_df.empty else 0
                    annot_max_time = annot_sub_df['End Time (s)'].max() if not annot_sub_df.empty else 0
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
        for _, row in pred_df.iterrows():
            class_name = row[self.column_name_predictions]
            begin_time = row["Begin Time (s)"]
            end_time = row["End Time (s)"]
            confidence = row["Confidence"]
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
        for _, row in annot_df.iterrows():
            class_name = row[self.column_name_annotations]
            begin_time = row["Begin Time (s)"]
            end_time = row["End Time (s)"]
            sample_indices = samples_df[
                (
                        samples_df["start_time"]
                        <= end_time - self.min_overlap
                )
                & (
                        samples_df["end_time"]
                        >= begin_time + self.min_overlap
                )
                ].index
            for i in sample_indices:
                samples_df.loc[i, f"{class_name}_annotation"] = 1

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
