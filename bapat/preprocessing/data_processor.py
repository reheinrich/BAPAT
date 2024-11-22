"""
DataProcessor class for handling and transforming sample data with annotations and predictions.

This module defines the DataProcessor class, which processes prediction and annotation data,
aligns them with sampled time intervals, and generates tensors for further model training or evaluation.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# from bapat.preprocessing.utils import (
#     extract_recording_filename,
#     extract_recording_filename_from_filename,
#     read_and_concatenate_files_in_directory,
# )

from preprocessing.utils import (
    extract_recording_filename,
    extract_recording_filename_from_filename,
    read_and_concatenate_files_in_directory,
)


class DataProcessor:
    """
    Processor for handling and transforming sample data with annotations and predictions.

    This class processes prediction and annotation data, aligning them with sampled time intervals,
    and generates tensors for further model training or evaluation.
    """

    # Default column mappings for predictions and annotations
    DEFAULT_COLUMNS_PREDICTIONS = {
        "Start Time": "Start Time",
        "End Time": "End Time",
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
        "Confidence": "Confidence",
    }

    DEFAULT_COLUMNS_ANNOTATIONS = {
        "Start Time": "Start Time",
        "End Time": "End Time",
        "Class": "Class",
        "Recording": "Recording",
        "Duration": "Duration",
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        prediction_directory_path: str,
        annotation_directory_path: str,
        prediction_file_name: Optional[str] = None,
        annotation_file_name: Optional[str] = None,
        class_mapping: Optional[Dict[str, str]] = None,
        sample_duration: int = 3,
        min_overlap: float = 0.5,
        columns_predictions: Optional[Dict[str, str]] = None,
        columns_annotations: Optional[Dict[str, str]] = None,
        recording_duration: Optional[float] = None,
    ) -> None:
        """
        Initializes the DataProcessor by loading prediction and annotation data.

        Args:
            prediction_directory_path (str): Path to the folder containing prediction files.
            prediction_file_name (Optional[str]): Name of the prediction file to process.
            annotation_directory_path (str): Path to the folder containing annotation files.
            annotation_file_name (Optional[str]): Name of the annotation file to process.
            class_mapping (Optional[Dict[str, str]]): Optional dictionary mapping raw class names to standardized class names.
            sample_duration (int, optional): Length of each data sample in seconds. Defaults to 3.
            min_overlap (float, optional): Minimum overlap required between prediction and annotation to consider a match.
            columns_predictions (Optional[Dict[str, str]], optional): Column name mappings for prediction files.
            columns_annotations (Optional[Dict[str, str]], optional): Column name mappings for annotation files.
            recording_duration (Optional[float], optional): User-specified recording duration in seconds. Defaults to None.
        """

        # Initialize instance variables
        self.sample_duration: int = sample_duration
        self.min_overlap: float = min_overlap
        self.class_mapping: Optional[Dict[str, str]] = class_mapping

        self.columns_predictions: Dict[str, str] = (
            columns_predictions
            if columns_predictions is not None
            else self.DEFAULT_COLUMNS_PREDICTIONS.copy()
        )
        self.columns_annotations: Dict[str, str] = (
            columns_annotations
            if columns_annotations is not None
            else self.DEFAULT_COLUMNS_ANNOTATIONS.copy()
        )

        self.recording_duration: Optional[float] = recording_duration

        # Paths and filenames
        self.prediction_directory_path: str = prediction_directory_path
        self.prediction_file_name: Optional[str] = prediction_file_name
        self.annotation_directory_path: str = annotation_directory_path
        self.annotation_file_name: Optional[str] = annotation_file_name

        # DataFrames for predictions and annotations
        self.predictions_df: pd.DataFrame = pd.DataFrame()
        self.annotations_df: pd.DataFrame = pd.DataFrame()

        # Placeholder for classes
        self.classes: Tuple[str, ...] = ()

        # Placeholder for samples DataFrame and tensors
        self.samples_df: pd.DataFrame = pd.DataFrame()
        self.prediction_tensors: np.ndarray = np.array([])
        self.label_tensors: np.ndarray = np.array([])

        # Ensure essential columns are provided and parameters are valid
        self._validate_columns()
        self._validate_parameters()

        # Load data
        self.load_data()

        # Process data
        self.process_data()

        # Create tensors
        self.create_tensors()

    def _validate_parameters(self):
        if self.sample_duration <= 0:
            raise ValueError("Sample duration must be positive")
        if not (0 <= self.min_overlap <= 1):
            raise ValueError("Min overlap must be between 0 and 1")
        if self.recording_duration is not None and self.recording_duration <= 0:
            raise ValueError("Recording duration must be positive")

    def _validate_columns(self) -> None:
        """
        Validates that the essential columns are provided in the column mappings.

        Raises:
            ValueError: If any required columns are missing or have a None value.
        """
        required_columns = ["Start Time", "End Time", "Class"]

        # Check for missing or None prediction columns
        missing_pred_columns = [
            col
            for col in required_columns
            if col not in self.columns_predictions
            or self.columns_predictions[col] is None
        ]

        # Check for missing or None annotation columns
        missing_annot_columns = [
            col
            for col in required_columns
            if col not in self.columns_annotations
            or self.columns_annotations[col] is None
        ]

        if missing_pred_columns:
            raise ValueError(
                f"Missing or None prediction columns: {', '.join(missing_pred_columns)}"
            )
        if missing_annot_columns:
            raise ValueError(
                f"Missing or None annotation columns: {', '.join(missing_annot_columns)}"
            )

    def load_data(self) -> None:
        """
        Loads the prediction and annotation data into DataFrames.

        Depending on whether specific files are provided, this method either reads all files
        in the given directories or reads the specified files.
        """
        if self.prediction_file_name is None or self.annotation_file_name is None:
            # Read all prediction files into a single DataFrame
            self.predictions_df = read_and_concatenate_files_in_directory(
                self.prediction_directory_path
            )
            # Read all annotation files into a single DataFrame
            self.annotations_df = read_and_concatenate_files_in_directory(
                self.annotation_directory_path
            )

            # Add source_file column if missing
            if "source_file" not in self.predictions_df.columns:
                self.predictions_df["source_file"] = ""

            if "source_file" not in self.annotations_df.columns:
                self.annotations_df["source_file"] = ""

            # Prepare DataFrames
            self.predictions_df = self._prepare_dataframe(
                self.predictions_df, prediction=True
            )

            self.annotations_df = self._prepare_dataframe(
                self.annotations_df, prediction=False
            )

            # Apply class mapping to prediction files only
            if self.class_mapping:
                class_col_pred = self.get_column_name("Class", prediction=True)
                self.predictions_df[class_col_pred] = self.predictions_df[
                    class_col_pred
                ].apply(lambda x: self.class_mapping.get(x, x))
        else:
            # Ensure that the prediction and annotation files match if specified
            if not self.prediction_file_name.startswith(
                os.path.splitext(self.annotation_file_name)[0]
            ):
                warnings.warn(
                    "Prediction file name and annotation file name do not fully match, but proceeding anyway."
                )

            # Process individual files
            prediction_file = os.path.join(
                self.prediction_directory_path, self.prediction_file_name
            )
            annotation_file = os.path.join(
                self.annotation_directory_path, self.annotation_file_name
            )

            self.predictions_df = pd.read_csv(prediction_file, sep="\t")
            self.annotations_df = pd.read_csv(annotation_file, sep="\t")

            # Add source_file column
            self.predictions_df["source_file"] = self.prediction_file_name
            self.annotations_df["source_file"] = self.annotation_file_name

            # Prepare DataFrames
            self.predictions_df = self._prepare_dataframe(
                self.predictions_df, prediction=True
            )

            self.annotations_df = self._prepare_dataframe(
                self.annotations_df, prediction=False
            )

            # Apply class mapping to prediction files only
            if self.class_mapping:
                class_col_pred = self.get_column_name("Class", prediction=True)
                self.predictions_df[class_col_pred] = self.predictions_df[
                    class_col_pred
                ].apply(lambda x: self.class_mapping.get(x, x))

        # Collect all classes from both prediction and annotation
        class_col_pred = self.get_column_name("Class", prediction=True)
        class_col_annot = self.get_column_name("Class", prediction=False)

        pred_classes = set(self.predictions_df[class_col_pred].unique())
        annot_classes = set(self.annotations_df[class_col_annot].unique())
        all_classes = pred_classes.union(annot_classes)
        self.classes = tuple(sorted(all_classes))

    def _prepare_dataframe(self, df: pd.DataFrame, prediction: bool) -> pd.DataFrame:
        """
        Prepares a DataFrame by adding 'recording_filename' column.

        This method extracts the recording filename from either a specified 'Recording' column
        or from the 'source_file' column.

        Args:
            df (pd.DataFrame): DataFrame to prepare.
            prediction (bool): Whether the DataFrame is for predictions or annotations.

        Returns:
            pd.DataFrame: The prepared DataFrame.
        """
        # If 'Recording' column is present, extract recording filename
        recording_col = self.get_column_name("Recording", prediction=prediction)
        if recording_col in df.columns:
            df["recording_filename"] = extract_recording_filename(df[recording_col])
        else:
            if "source_file" in df.columns:
                df["recording_filename"] = extract_recording_filename_from_filename(
                    df["source_file"]
                )
            else:
                df["recording_filename"] = ""

        return df

    def process_data(self) -> None:
        """
        Processes the loaded data, aligns predictions and annotations with samples,
        and updates the samples DataFrame.

        This method iterates over all recording filenames, processes each recording,
        and accumulates the results in the samples DataFrame.
        """
        self.samples_df = pd.DataFrame()

        # Get the set of all recording filenames
        recording_filenames = set(
            self.predictions_df["recording_filename"].unique()
        ).union(set(self.annotations_df["recording_filename"].unique()))

        # Process each recording
        for recording_filename in recording_filenames:
            pred_df = self.predictions_df[
                self.predictions_df["recording_filename"] == recording_filename
            ]
            annot_df = self.annotations_df[
                self.annotations_df["recording_filename"] == recording_filename
            ]

            # Process the recording
            samples_df = self.process_recording(recording_filename, pred_df, annot_df)

            # Append samples_df to self.samples_df
            self.samples_df = pd.concat(
                [self.samples_df, samples_df], ignore_index=True
            )

    def process_recording(
        self, recording_filename: str, pred_df: pd.DataFrame, annot_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes a single recording by determining file duration, initializing samples,
        and updating samples with predictions and annotations.

        Args:
            recording_filename (str): The name of the recording.
            pred_df (pd.DataFrame): Predictions DataFrame for the recording.
            annot_df (pd.DataFrame): Annotations DataFrame for the recording.

        Returns:
            pd.DataFrame: DataFrame of samples for the recording.
        """
        # Determine file duration
        file_duration = self.determine_file_duration(pred_df, annot_df)

        if file_duration <= 0:
            return (
                pd.DataFrame()
            )  # Return empty DataFrame if duration is zero or negative

        # Initialize samples
        samples_df = self.initialize_samples(
            recording_filename=recording_filename, file_duration=file_duration
        )

        # Update samples_df with predictions and annotations
        self.update_samples_with_predictions(pred_df, samples_df)
        self.update_samples_with_annotations(annot_df, samples_df)

        return samples_df

    def determine_file_duration(
        self, pred_df: pd.DataFrame, annot_df: pd.DataFrame
    ) -> float:
        """
        Determine the duration of the recording based on the dataframes and recording_duration.

        Args:
            pred_df (pd.DataFrame): Predictions DataFrame.
            annot_df (pd.DataFrame): Annotations DataFrame.

        Returns:
            float: The duration of the recording.
        """
        if self.recording_duration is not None:
            return self.recording_duration

        duration = 0.0

        # Check in predictions with column mapping
        file_duration_col_pred = self.get_column_name("Duration", prediction=True)

        file_duration_col_annot = self.get_column_name("Duration", prediction=False)

        # Try to get duration from 'Duration' column in pred_df
        if (
            file_duration_col_pred in pred_df.columns
            and pred_df[file_duration_col_pred].notnull().any()
        ):
            duration = max(duration, pred_df[file_duration_col_pred].dropna().max())

        # Try to get duration from 'Duration' column in annot_df
        if (
            file_duration_col_annot in annot_df.columns
            and annot_df[file_duration_col_annot].notnull().any()
        ):
            duration = max(duration, annot_df[file_duration_col_annot].dropna().max())

        # If 'Duration' not available, use max 'End Time'
        if duration == 0.0:
            end_time_col_pred = self.get_column_name("End Time", prediction=True)
            end_time_col_annot = self.get_column_name("End Time", prediction=False)

            max_end_pred = (
                pred_df[end_time_col_pred].max()
                if end_time_col_pred in pred_df.columns
                else 0.0
            )
            max_end_annot = (
                annot_df[end_time_col_annot].max()
                if end_time_col_annot in annot_df.columns
                else 0.0
            )
            duration = max(max_end_pred, max_end_annot)

            # Handle NaN or negative values
            if pd.isna(duration) or duration < 0:
                duration = 0.0

        return duration

    def initialize_samples(
        self, recording_filename: str, file_duration: float
    ) -> pd.DataFrame:
        """
        Initializes a DataFrame of time-based sample intervals using the file duration and sample duration.

        Each sample represents a time interval of length 'sample_duration', and samples are created
        to cover the entire recording duration.

        Args:
            recording_filename (str): The name of the recording.
            file_duration (float): The total duration of the file in seconds.

        Returns:
            pd.DataFrame: DataFrame containing initialized sample intervals, confidence scores, and annotations.
        """
        if file_duration <= 0:
            return pd.DataFrame()

        # Create intervals that ensure each sample has the exact sample duration
        intervals = np.arange(0, file_duration, self.sample_duration)
        if len(intervals) == 0:
            intervals = np.array([0])

        samples = {
            "filename": recording_filename,
            "sample_index": [],
            "start_time": [],
            "end_time": [],
        }

        for idx, start in enumerate(intervals):
            samples["sample_index"].append(idx)
            samples["start_time"].append(start)
            samples["end_time"].append(min(start + self.sample_duration, file_duration))

        # Initialize columns for confidence scores and annotations for each class
        for label in self.classes:
            samples[f"{label}_confidence"] = [0.0] * len(
                samples["sample_index"]
            )  # Float values
            samples[f"{label}_annotation"] = [0] * len(
                samples["sample_index"]
            )  # Integer values

        return pd.DataFrame(samples)

    def update_samples_with_predictions(
        self, pred_df: pd.DataFrame, samples_df: pd.DataFrame
    ) -> None:
        """
        Updates the samples DataFrame with prediction confidence scores from the predictions DataFrame.

        For each prediction, the method finds overlapping samples based on 'min_overlap' and updates
        the confidence scores. If multiple predictions overlap a single sample, it retains the maximum confidence.

        Args:
            pred_df (pd.DataFrame): Predictions DataFrame for the recording.
            samples_df (pd.DataFrame): Samples DataFrame to update.
        """
        class_col = self.get_column_name("Class", prediction=True)
        start_time_col = self.get_column_name("Start Time", prediction=True)
        end_time_col = self.get_column_name("End Time", prediction=True)
        confidence_col = self.get_column_name("Confidence", prediction=True)

        for _, row in pred_df.iterrows():
            class_name = row[class_col]
            if class_name not in self.classes:
                continue  # Skip classes not in the predefined list
            begin_time = row[start_time_col]
            end_time = row[end_time_col]
            confidence = row.get(confidence_col, 0.0)

            # Find samples that overlap with the prediction considering min_overlap
            sample_indices = samples_df[
                (samples_df["start_time"] <= end_time - self.min_overlap)
                & (samples_df["end_time"] >= begin_time + self.min_overlap)
            ].index

            # Update the confidence scores for overlapping samples
            for i in sample_indices:
                current_confidence = samples_df.loc[i, f"{class_name}_confidence"]
                samples_df.loc[i, f"{class_name}_confidence"] = max(
                    current_confidence, confidence
                )

    def update_samples_with_annotations(
        self, annot_df: pd.DataFrame, samples_df: pd.DataFrame
    ) -> None:
        """
        Updates the samples DataFrame with annotations from the annotations DataFrame.

        For each annotation, the method finds overlapping samples based on 'min_overlap' and sets
        the annotation value to 1 for those samples.

        Args:
            annot_df (pd.DataFrame): Annotations DataFrame for the recording.
            samples_df (pd.DataFrame): Samples DataFrame to update.
        """
        class_col = self.get_column_name("Class", prediction=False)
        start_time_col = self.get_column_name("Start Time", prediction=False)
        end_time_col = self.get_column_name("End Time", prediction=False)

        for _, row in annot_df.iterrows():
            class_name = row[class_col]
            if class_name not in self.classes:
                continue  # Skip classes not in the predefined list
            begin_time = row[start_time_col]
            end_time = row[end_time_col]

            # Find samples that overlap with the annotation considering min_overlap
            sample_indices = samples_df[
                (samples_df["start_time"] <= end_time - self.min_overlap)
                & (samples_df["end_time"] >= begin_time + self.min_overlap)
            ].index

            # Assign a value of 1 to the overlapping samples
            for i in sample_indices:
                samples_df.loc[i, f"{class_name}_annotation"] = 1

    def create_tensors(self):
        """Creates numpy arrays from the samples DataFrame."""
        if self.samples_df.empty:
            self.prediction_tensors = np.empty((0, len(self.classes)), dtype=np.float32)
            self.label_tensors = np.empty((0, len(self.classes)), dtype=np.int64)
            return

        # Check for NaN values in annotation columns
        annotation_columns = [f"{cls}_annotation" for cls in self.classes]
        if self.samples_df[annotation_columns].isnull().values.any():
            raise ValueError("NaN values found in annotation columns.")

        # Check for NaN values in confidence columns
        confidence_columns = [f"{cls}_confidence" for cls in self.classes]
        if self.samples_df[confidence_columns].isnull().values.any():
            raise ValueError("NaN values found in confidence columns.")

        # Create numpy arrays
        self.prediction_tensors = self.samples_df[confidence_columns].to_numpy(
            dtype=np.float32
        )
        self.label_tensors = self.samples_df[annotation_columns].to_numpy(
            dtype=np.int64
        )

    def get_column_name(self, field_name: str, prediction: bool = True) -> str:
        """Get the column name from the appropriate mapping."""
        if field_name is None:
            raise TypeError("field_name cannot be None.")
        if prediction is None:
            raise TypeError("prediction parameter cannot be None.")

        mapping = self.columns_predictions if prediction else self.columns_annotations

        if field_name in mapping and mapping[field_name] is not None:
            return mapping[field_name]

        return field_name

    def get_sample_data(self) -> pd.DataFrame:
        """
        Retrieves the DataFrame containing all the sample intervals, prediction scores, and annotations.

        Returns:
            pd.DataFrame: The DataFrame representing sampled data.
        """
        return self.samples_df.copy()

    def get_filtered_tensors(
        self,
        selected_classes: Optional[List[str]] = None,
        selected_recordings: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[str]]:
        """
        Filters the tensors based on selected classes and recordings.

        Parameters:
            selected_classes (List[str], optional): Classes to include.
            selected_recordings (List[str], optional): Recordings to include.

        Returns:
            Tuple containing filtered prediction tensors, label tensors, and classes.
        """
        if self.samples_df.empty:
            raise ValueError("samples_df is empty.")

        if "filename" not in self.samples_df.columns:
            raise ValueError("samples_df must contain a 'filename' column.")

        classes = (
            self.classes
            if selected_classes is None
            else tuple(cls for cls in selected_classes if cls in self.classes)
        )

        if not classes:
            raise ValueError("No valid classes selected.")

        mask = pd.Series(True, index=self.samples_df.index)

        if selected_recordings is not None:
            if selected_recordings:
                mask &= self.samples_df["filename"].isin(selected_recordings)
            else:
                # If selected_recordings is an empty list, select no samples
                mask = pd.Series(False, index=self.samples_df.index)

        filtered_samples = self.samples_df.loc[mask]

        confidence_columns = [f"{cls}_confidence" for cls in classes]
        annotation_columns = [f"{cls}_annotation" for cls in classes]

        if not all(
            col in filtered_samples.columns
            for col in confidence_columns + annotation_columns
        ):
            raise KeyError("Required confidence or annotation columns are missing.")

        predictions = filtered_samples[confidence_columns].to_numpy(dtype=np.float32)
        labels = filtered_samples[annotation_columns].to_numpy(dtype=np.int64)

        return predictions, labels, classes
