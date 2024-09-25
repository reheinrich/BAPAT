import sys
import os
import json

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QTextEdit,
    QSpinBox, QDoubleSpinBox,
    QToolButton, QGroupBox, QCheckBox, QLineEdit, QSizePolicy, QFormLayout, QGridLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor, QIcon
from PySide6.QtWidgets import QStyle

import matplotlib
matplotlib.use('QtAgg')  # Set the Matplotlib backend to QtAgg

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Import your existing classes
from preprocessing.data_processing import DataProcessor
from assessment.performance_assessor import PerformanceAssessor

# Disable interactive mode to avoid automatic figure creation
plt.ioff()  # Add this line to disable the interactive mode

class DropLineEdit(QLineEdit):
    """Custom QLineEdit that accepts drag-and-drop of files and folders."""
    def __init__(self, parent=None):
        super(DropLineEdit, self).__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super(DropLineEdit, self).dragEnterEvent(event)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path):
                self.setText(file_path)
            elif os.path.isfile(file_path):
                self.setText(file_path)
        event.acceptProposedAction()


class PerformanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Performance Assessor")
        self.resize(900, 700)

        # Track previous state
        self.previous_annotation_path = None
        self.previous_prediction_path = None
        self.previous_mapping_path = None
        self.previous_sample_duration = None
        self.previous_min_overlap = None
        self.previous_column_name_predictions = None
        self.previous_column_name_annotations = None
        self.processor = None

        self.init_ui()

    def init_ui(self):
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # File selection layout
        self.file_selection_group = QGroupBox("File Selection")
        self.file_selection_layout = QFormLayout()
        self.file_selection_layout.setSpacing(10)
        self.file_selection_layout.setContentsMargins(10, 10, 10, 10)

        # Create a horizontal layout for predictions and annotations with spacing
        file_inputs_layout = QHBoxLayout()
        file_inputs_layout.setSpacing(20)  # Add horizontal space between inputs

        # Annotations folder selection
        annotation_tooltip = "Select the folder that contains the true labels for evaluation."
        self.annotation_input = DropLineEdit()
        self.annotation_button = QPushButton("Browse")
        self.annotation_button.clicked.connect(self.select_annotation_folder)
        # New QLineEdit for annotation column name
        self.annotation_column_input = QLineEdit()
        self.annotation_column_input.setFixedWidth(150)
        self.annotation_column_input.setPlaceholderText("Class Column Name")
        self.annotation_column_input.setToolTip(
            "Specify the name of the column in your annotation files that contains the relevant annotations. The default is 'Class'.")
        self.annotation_column_input.setStyleSheet("""
                    QLineEdit {
                        color: white;
                    }
                    QLineEdit::placeholder {
                        color: white;
                    }
                    QToolTip {
                        background-color: #2C3E50;  /* Darker background */
                        color: white;  /* White text */
                        border: 1px solid #1E3A5F;
                    }
                """)
        self.add_file_input("Annotations Folder:", self.annotation_input, self.annotation_button, annotation_tooltip,
                            file_inputs_layout, extra_widget=self.annotation_column_input)

        #self.add_file_input("Annotations Folder:", self.annotation_input, self.annotation_button, annotation_tooltip, file_inputs_layout)

        # Predictions folder selection
        prediction_tooltip = "Select the folder that contains the model's prediction files."
        self.prediction_input = DropLineEdit()
        self.prediction_button = QPushButton("Browse")
        self.prediction_button.clicked.connect(self.select_prediction_folder)
        # New QLineEdit for prediction column name
        self.prediction_column_input = QLineEdit()
        self.prediction_column_input.setFixedWidth(150)
        self.prediction_column_input.setPlaceholderText("Class Column Name")
        self.prediction_column_input.setToolTip(
            "Specify the name of the column in your prediction files that contains the class names. The default is 'Common Name'.")
        self.prediction_column_input.setStyleSheet("""
                            QLineEdit {
                                color: white;
                            }
                            QLineEdit::placeholder {
                                color: white;
                            }
                            QToolTip {
                                background-color: #2C3E50;  /* Darker background */
                                color: white;  /* White text */
                                border: 1px solid #1E3A5F;
                            }
                        """)
        self.add_file_input("Predictions Folder:", self.prediction_input, self.prediction_button, prediction_tooltip,
                            file_inputs_layout, extra_widget=self.prediction_column_input)
        #self.add_file_input("Predictions Folder:", self.prediction_input, self.prediction_button, prediction_tooltip, file_inputs_layout)

        self.file_selection_layout.addRow(file_inputs_layout)

        # Class mapping file selection (optional)
        class_mapping_tooltip = "Optional: Select a JSON file that maps class names between your prediction and annotation files."
        self.mapping_input = DropLineEdit()
        self.mapping_input.setFixedWidth(300)  # Make it smaller
        self.mapping_button = QPushButton("Browse")
        self.mapping_button.clicked.connect(self.select_mapping_file)

        # Create the download button
        self.download_mapping_button = QPushButton("Download Template")
        self.download_mapping_button.setFixedWidth(150)
        self.download_mapping_button.clicked.connect(self.download_class_mapping_template)
        self.download_mapping_button.setToolTip(
            "Click to download a template JSON file to map class names between predictions and annotations. Use this if class names differ between your prediction and annotation files.")
        self.download_mapping_button.setStyleSheet("""
            QToolTip {
                background-color: #2C3E50;  /* Darker background */
                color: white;  /* White text */
                border: 1px solid #1E3A5F;
            }
        """)

        # Center the class mapping input
        mapping_layout = QHBoxLayout()
        mapping_layout.addWidget(self.download_mapping_button)
        mapping_layout.addStretch()
        label = QLabel("Class Mapping (Optional):")
        label.setAlignment(Qt.AlignCenter)
        tooltip_button = QToolButton()
        tooltip_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        tooltip_button.setToolTip(class_mapping_tooltip)
        tooltip_button.setStyleSheet("""
            QToolButton:hover {
                background-color: #294C7D;  /* Lighter blue when hovered */
            }
            QToolTip {
                background-color: #2C3E50;  /* Darker background for the tooltip */
                color: white;  /* White text for the tooltip */
                border: 1px solid #1E3A5F;
            }
        """)
        mapping_layout.addWidget(label)
        mapping_layout.addWidget(tooltip_button)
        mapping_layout.addWidget(self.mapping_input)
        mapping_layout.addWidget(self.mapping_button)
        mapping_layout.addStretch()
        self.file_selection_layout.addRow(mapping_layout)

        self.file_selection_group.setLayout(self.file_selection_layout)
        self.layout.addWidget(self.file_selection_group)

        # Parameters and Metrics layout
        parameters_and_metrics_layout = QHBoxLayout()

        # Parameters group
        self.parameters_group = QGroupBox("Parameters")
        parameters_layout = QGridLayout()
        parameters_layout.setSpacing(10)
        parameters_layout.setContentsMargins(10, 10, 10, 10)

        row = 0
        # Sample duration
        self.sample_duration_spin = QSpinBox()
        self.sample_duration_spin.setValue(3)
        self.sample_duration_spin.setMinimum(1)
        sample_duration_tooltip = "Set the length of each audio sample in seconds."
        self.add_parameter_input(row, "Sample Duration (s):", self.sample_duration_spin, sample_duration_tooltip,
                                 parameters_layout)
        row += 1

        # Minimum overlap
        self.min_overlap_spin = QDoubleSpinBox()
        self.min_overlap_spin.setValue(0.5)
        self.min_overlap_spin.setSingleStep(0.1)
        self.min_overlap_spin.setMinimum(0.0)
        min_overlap_tooltip = "Specify the minimum required overlap (in seconds) between an annotation and a sample for the annotation to be considered for that sample."

        self.add_parameter_input(row, "Minimum Overlap (s):", self.min_overlap_spin, min_overlap_tooltip,
                                 parameters_layout)
        row += 1

        # Threshold slider and display
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(99)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.threshold_changed)
        self.threshold_value_label = QLabel(f"{self.threshold_slider.value() / 100:.2f}")
        self.threshold_value_label.setFixedWidth(40)
        threshold_tooltip = "Adjust the threshold value for classifying a prediction as positive."
        self.add_parameter_input(row, "Threshold:", self.threshold_slider, threshold_tooltip, parameters_layout,
                                 extra_widget=self.threshold_value_label)
        row += 1

        # Class-wise Metrics checkbox
        self.class_wise_checkbox = QCheckBox()
        self.class_wise_checkbox.setChecked(False)
        class_wise_tooltip = "Check this box to see performance metrics for each class individually."
        self.add_parameter_input(row, "Class-wise Metrics:", self.class_wise_checkbox, class_wise_tooltip, parameters_layout)
        row += 1

        self.parameters_group.setLayout(parameters_layout)

        parameters_and_metrics_layout.addWidget(self.parameters_group)

        # Buttons vertical layout
        buttons_group = QGroupBox()
        buttons_group_layout = QVBoxLayout()
        buttons_group_layout.setSpacing(40)  # Increased vertical spacing between buttons
        buttons_group_layout.setContentsMargins(10, 10, 10, 10)
        buttons_group_layout.addStretch()
        buttons_group.setLayout(buttons_group_layout)

        # Define button size
        button_width = 200
        button_height = 50

        # Calculate Metrics button
        self.calculate_button = QPushButton("Calculate Metrics")
        self.calculate_button.clicked.connect(self.calculate_metrics)
        self.calculate_button.setFixedSize(button_width, button_height)
        buttons_group_layout.addWidget(self.calculate_button, alignment=Qt.AlignCenter)

        # Plot Metrics button
        self.plot_metrics_button = QPushButton("Plot Metrics")
        self.plot_metrics_button.clicked.connect(self.plot_metrics)
        self.plot_metrics_button.setFixedSize(button_width, button_height)
        buttons_group_layout.addWidget(self.plot_metrics_button, alignment=Qt.AlignCenter)

        # Plot Confusion Matrix button
        self.plot_confusion_button = QPushButton("Plot Confusion Matrix")
        self.plot_confusion_button.clicked.connect(self.plot_confusion_matrix)
        self.plot_confusion_button.setToolTip(
            "Click to display a confusion matrix showing the percentage of correct and incorrect predictions made by the model for each class.")
        self.plot_confusion_button.setStyleSheet("""
            QToolTip {
                background-color: #2C3E50;
                color: white;
                border: 1px solid #1E3A5F;
            }
        """)
        self.plot_confusion_button.setFixedSize(button_width, button_height)
        buttons_group_layout.addWidget(self.plot_confusion_button, alignment=Qt.AlignCenter)

        # Plot Metrics All Thresholds button
        self.plot_metrics_all_thresholds_button = QPushButton("Plot Metrics All Thresholds")
        self.plot_metrics_all_thresholds_button.clicked.connect(self.plot_metrics_all_thresholds)
        self.plot_metrics_all_thresholds_button.setFixedSize(button_width, button_height)

        buttons_group_layout.addWidget(self.plot_metrics_all_thresholds_button, alignment=Qt.AlignCenter)

        buttons_group_layout.addStretch()

        # Add the buttons group to the parameters and metrics layout
        parameters_and_metrics_layout.addWidget(buttons_group)

        # Metrics group
        self.metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(10)
        metrics_layout.setContentsMargins(10, 10, 10, 10)

        # Define the metrics and their descriptions
        metric_info = {
            "AUROC": """<b>AUROC</b><br>
            AUROC measures the probability that the model will rank a random positive case higher than a random negative case.<br>
            Advantage: It provides an overall sense of the model's ability to correctly discriminate between classes across all thresholds and is particularly useful in cases of class imbalance.<br>
            Disadvantage: It may be difficult to interpret.""",
            "Precision": """<b>Precision</b><br>
            Precision measures how often the model's positive predictions are actually correct.<br>
            Advantage: It highlights the model's ability to make accurate predictions, reducing unnecessary false alarms.<br>
            Disadvantage: It doesn't show how many positive cases the model missed, which can be problematic if missing positive cases is costly.""",
            "Recall": """<b>Recall</b><br>
            Recall measures the percentage of positive cases that the model successfully identifies for each class.<br>
            Advantage: It ensures that the model doesn't miss any positive cases, which is critical when every positive case counts.<br>
            Disadvantage: It doesn't take into account how many wrong guesses the model makes, which can lead to many false alarms.""",
            "F1 Score": """<b>F1 Score</b><br>
            The F1 score is the harmonic mean of precision and recall, providing a balance between the two.<br>
            Advantage: It provides a balanced metric that gives equal weight to correct predictions and missed positive cases.<br>
            Disadvantage: It can be less intuitive to interpret and may not reflect true performance if precision and recall are very different.""",
            "Average Precision (AP)": """<b>Average Precision (AP)</b><br>
            Average Precision summarizes the precision-recall curve by averaging the precision values across all recall levels.<br>
            Advantage: AP provides a single number that represents the performance of the model across all thresholds.<br>
            Disadvantage: It can be noisy and less reliable for classes with very few positive cases and difficult to interpret.""",
            "Accuracy": """<b>Accuracy</b><br>
            Accuracy measures the percentage of times the model correctly predicts the correct class.<br>
            Advantage: It's a simple way to measure how often the model is correct across all classes equally.<br>
            Disadvantage: It can be misleading if some classes have many more examples than others, or if the model's errors are significant.""",
        }

        self.metrics_checkboxes = {}

        for metric_name, description in metric_info.items():
            # Create a horizontal layout for each metric
            metric_layout = QHBoxLayout()
            # Checkbox
            checkbox = QCheckBox(metric_name)
            checkbox.setChecked(True)  # Default to selected
            self.metrics_checkboxes[metric_name.lower()] = checkbox
            metric_layout.addWidget(checkbox)
            # Question mark button with tooltip
            tooltip_button = QToolButton()
            tooltip_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
            tooltip_button.setToolTip(description)
            tooltip_button.setStyleSheet("""
                QToolButton:hover {
                    background-color: #294C7D;  /* Lighter blue when hovered */
                }
                QToolTip {
                    background-color: #2C3E50;  /* Darker background for the tooltip */
                    color: white;  /* White text for the tooltip */
                    border: 1px solid #1E3A5F;
                }
            """)
            metric_layout.addWidget(tooltip_button)
            metrics_layout.addLayout(metric_layout)

        self.metrics_group.setLayout(metrics_layout)
        parameters_and_metrics_layout.addWidget(self.metrics_group)

        # Set stretch factors to divide the horizontal space equally
        parameters_and_metrics_layout.setStretch(0, 1)  # Parameters group
        parameters_and_metrics_layout.setStretch(1, 1)  # Buttons group
        parameters_and_metrics_layout.setStretch(2, 1)  # Metrics group

        self.layout.addLayout(parameters_and_metrics_layout)

        # Download buttons layout
        download_buttons_layout = QHBoxLayout()

        # Download Results Table button
        self.download_results_button = QPushButton("Download Results Table")
        self.download_results_button.clicked.connect(self.download_results_table)
        self.download_results_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        download_buttons_layout.addWidget(self.download_results_button)

        # Download Data Table button
        self.download_data_button = QPushButton("Download Data Table")
        self.download_data_button.clicked.connect(self.download_data_table)
        self.download_data_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        download_buttons_layout.addWidget(self.download_data_button)

        # Add the download buttons layout to the main layout
        self.layout.addLayout(download_buttons_layout)

        # Ensure all buttons have the same fixed height
        button_height = self.calculate_button.sizeHint().height()
        self.calculate_button.setFixedHeight(button_height)
        self.plot_metrics_button.setFixedHeight(button_height)
        self.plot_confusion_button.setFixedHeight(button_height)
        self.download_results_button.setFixedHeight(button_height)
        self.download_data_button.setFixedHeight(button_height)
        self.plot_metrics_all_thresholds_button.setFixedHeight(button_height)

        # Results area
        self.results_group = QGroupBox("Results")
        self.results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_layout.addWidget(self.results_text)

        # Matplotlib figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        plt.close('all')

        self.results_group.setLayout(self.results_layout)
        self.layout.addWidget(self.results_group)

        # Adjust layout to make results field larger
        self.layout.setStretchFactor(self.results_group, 1)
        self.layout.setStretchFactor(parameters_and_metrics_layout, 0)
        self.layout.setStretchFactor(self.file_selection_group, 0)

    def add_file_input(self, label_text, line_edit, button, tooltip_text, parent_layout=None, extra_widget=None):
        layout = QVBoxLayout()
        label_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        label_layout.addWidget(label)
        # Add question mark icon with tooltip
        tooltip_button = QToolButton()
        tooltip_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        tooltip_button.setToolTip(tooltip_text)
        tooltip_button.setStyleSheet("""
            QToolButton:hover {
                background-color: #294C7D;
            }
            QToolTip {
                background-color: #2C3E50;
                color: white;
                border: 1px solid #1E3A5F;
            }
        """)
        label_layout.addWidget(tooltip_button)
        # Add the extra widget (QLineEdit for column name)
        if extra_widget:
            label_layout.addWidget(extra_widget)
        label_layout.addStretch()
        layout.addLayout(label_layout)
        line_edit.setFixedHeight(50)
        layout.addWidget(line_edit)
        button.setFixedHeight(30)
        layout.addWidget(button)
        if parent_layout:
            parent_layout.addLayout(layout)
        else:
            self.file_selection_layout.addRow(layout)

    def add_parameter_input(self, row, label_text, widget, tooltip_text, parent_layout, extra_widget=None):
        label = QLabel(label_text)
        tooltip_button = QToolButton()
        tooltip_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        tooltip_button.setToolTip(tooltip_text)
        tooltip_button.setStyleSheet("""
            QToolButton:hover {
                background-color: #294C7D;  /* Lighter blue when hovered */
            }
            QToolTip {
                background-color: #2C3E50;  /* Darker background for the tooltip */
                color: white;  /* White text for the tooltip */
                border: 1px solid #1E3A5F;
            }
        """)
        parent_layout.addWidget(label, row, 0)
        parent_layout.addWidget(tooltip_button, row, 1)
        parent_layout.addWidget(widget, row, 2)
        if extra_widget:
            parent_layout.addWidget(extra_widget, row, 3)
        else:
            spacer = QWidget()
            spacer.setFixedWidth(40)  # Adjust as needed
            parent_layout.addWidget(spacer, row, 3)

    def select_annotation_folder(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog

            # Allow the user to select either a folder or a .txt file
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Annotation File or Folder", "",
                                                       "Text Files (*.txt);;All Files (*)", options=options)
            if not file_name:
                folder = QFileDialog.getExistingDirectory(self, "Select Annotation Folder", options=options)
                if folder:
                    self.annotation_input.setText(folder)
            else:
                self.annotation_input.setText(file_name)
            self.reset_results()
        except Exception as e:
            print(f"Error selecting annotation file or folder: {e}")

    def select_prediction_folder(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog

            # Allow the user to select either a folder or a .txt file
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Prediction File or Folder", "",
                                                       "Text Files (*.txt);;All Files (*)", options=options)
            if not file_name:
                folder = QFileDialog.getExistingDirectory(self, "Select Prediction Folder", options=options)
                if folder:
                    self.prediction_input.setText(folder)
            else:
                self.prediction_input.setText(file_name)
            self.reset_results()
        except Exception as e:
            print(f"Error selecting prediction file or folder: {e}")

    def select_mapping_file(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Class Mapping File", "", "JSON Files (*.json)", options=options)
            if file_name:
                self.mapping_input.setText(file_name)
                self.reset_results()
        except Exception as e:
            print(f"Error selecting mapping file: {e}")

    def threshold_changed(self):
        self.current_threshold = self.threshold_slider.value() / 100.0
        self.threshold_value_label.setText(f"{self.current_threshold:.2f}")
        # No automatic update to results when threshold changes

    def calculate_metrics(self):
        if not self.update_processor_and_performance_assessor():
            return  # Exit if updating parameters failed

        # Calculate metrics
        self.display_metrics()


    def display_metrics(self):
        threshold = self.threshold_slider.value() / 100.0
        self.pa.threshold = threshold

        per_class = self.class_wise_checkbox.isChecked()

        # Calculate metrics
        metrics_df = self.pa.calculate_metrics(self.predictions, self.labels, per_class_metrics=per_class)
        self.results_text.setText(metrics_df.to_string())

    def plot_metrics(self):
        if not self.update_processor_and_performance_assessor():
            return  # Exit if updating parameters failed

        per_class = self.class_wise_checkbox.isChecked()

        # Calculate metrics
        metrics_df = self.pa.calculate_metrics(self.predictions, self.labels, per_class_metrics=per_class)
        self.results_text.setText(metrics_df.to_string())

        # Plot metrics
        self.figure.clear()
        self.pa.plot_metrics(self.predictions, self.labels, per_class_metrics=per_class)
        self.canvas.draw()

    def plot_confusion_matrix(self):
        if not self.update_processor_and_performance_assessor():
            return  # Exit if updating parameters failed

        # Plot confusion matrix
        self.figure.clear()
        self.pa.plot_confusion_matrix(self.predictions, self.labels)
        self.canvas.draw()

    def plot_metrics_all_thresholds(self):
        if not self.update_processor_and_performance_assessor():
            return  # Exit if updating parameters failed

        per_class = self.class_wise_checkbox.isChecked()

        # Plot metrics across thresholds
        self.figure.clear()
        self.pa.plot_metrics_all_thresholds(self.predictions, self.labels, per_class_metrics=per_class)
        self.canvas.draw()

    def update_processor_and_performance_assessor(self):
        # Get current input values
        annotation_path = self.annotation_input.text()
        prediction_path = self.prediction_input.text()
        mapping_path = self.mapping_input.text()
        sample_duration = self.sample_duration_spin.value()
        min_overlap = self.min_overlap_spin.value()
        column_name_predictions = self.prediction_column_input.text() or "Common Name"
        column_name_annotations = self.annotation_column_input.text() or "Class"

        # Check if annotation or prediction paths are empty
        if not annotation_path or not prediction_path:
            self.results_text.setText("Please select both annotation and prediction folders.")
            return False  # Exit if required paths are not provided

        # Load class mapping if provided
        class_mapping = None
        if mapping_path:
            try:
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
            except Exception as e:
                self.results_text.setText(f"Error loading class mapping file: {e}")
                return False  # Exit if loading the mapping fails

        # Check if the paths refer to files or directories
        annotation_dir, annotation_file = (os.path.dirname(annotation_path), os.path.basename(annotation_path)) \
            if os.path.isfile(annotation_path) else (annotation_path, None)

        prediction_dir, prediction_file = (os.path.dirname(prediction_path), os.path.basename(prediction_path)) \
            if os.path.isfile(prediction_path) else (prediction_path, None)

        # Determine if reinitialization is necessary
        if (self.processor is None or
                self.previous_annotation_path != annotation_path or
                self.previous_prediction_path != prediction_path or
                self.previous_mapping_path != mapping_path or
                self.previous_sample_duration != sample_duration or
                self.previous_min_overlap != min_overlap or
                self.previous_column_name_predictions != column_name_predictions or
                self.previous_column_name_annotations != column_name_annotations):

            try:
                # Initialize DataProcessor
                self.processor = DataProcessor(
                    prediction_directory_path=prediction_dir,
                    prediction_file_name=prediction_file,
                    annotation_directory_path=annotation_dir,
                    annotation_file_name=annotation_file,
                    class_mapping=class_mapping,
                    sample_duration=sample_duration,
                    min_overlap=min_overlap,
                    column_name_predictions=column_name_predictions,
                    column_name_annotations=column_name_annotations
                )

                # Update the stored parameters after initialization
                self.previous_annotation_path = annotation_path
                self.previous_prediction_path = prediction_path
                self.previous_mapping_path = mapping_path
                self.previous_sample_duration = sample_duration
                self.previous_min_overlap = min_overlap
                self.previous_column_name_predictions = column_name_predictions
                self.previous_column_name_annotations = column_name_annotations

            except Exception as e:
                self.results_text.setText(f"Error initializing DataProcessor: {e}")
                return False  # Exit if initialization fails

        # Get predictions and labels from the processor
        self.predictions = self.processor.get_prediction_tensor()
        self.labels = self.processor.get_label_tensor()
        classes = self.processor.classes
        num_classes = len(classes)

        # Determine the task type (binary or multilabel)
        task = 'binary' if num_classes == 1 else 'multilabel'

        # Extract selected metrics
        selected_metrics = []
        valid_metrics = {
            "accuracy": "accuracy",
            "recall": "recall",
            "precision": "precision",
            "f1 score": "f1",
            "average precision (ap)": "ap",
            "auroc": "auroc"
        }
        for metric_name_lower, checkbox in self.metrics_checkboxes.items():
            if checkbox.isChecked():
                selected_metrics.append(valid_metrics[metric_name_lower])

        metrics = tuple(selected_metrics)

        # Initialize PerformanceAssessor
        self.pa = PerformanceAssessor(
            num_classes=num_classes,
            threshold=self.threshold_slider.value() / 100.0,
            classes=classes,
            task=task,
            metrics=metrics
        )

        return True  # Indicate successful update

    def download_results_table(self):
        if not hasattr(self, 'pa'):
            self.results_text.setText("Please calculate metrics first.")
            return

        per_class = self.class_wise_checkbox.isChecked()
        metrics_df = self.pa.calculate_metrics(self.predictions, self.labels, per_class_metrics=per_class)

        # Open a file dialog to select the save location
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Results Table", "results.csv", "CSV Files (*.csv)", options=options)
        if file_name:
            try:
                metrics_df.to_csv(file_name, index=True)
                self.results_text.setText(f"Results table saved to {file_name}")
            except Exception as e:
                self.results_text.setText(f"Error saving results table: {e}")

    def download_data_table(self):
        if not hasattr(self, 'processor'):
            self.results_text.setText("Please process data first.")
            return

        try:
            data_df = self.processor.get_sample_data()
            # Open a file dialog to select the save location
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Data Table", "data.csv", "CSV Files (*.csv)",
                                                       options=options)
            if file_name:
                data_df.to_csv(file_name, index=False)
                self.results_text.setText(f"Data table saved to {file_name}")
        except Exception as e:
            self.results_text.setText(f"Error saving data table: {e}")

    def download_class_mapping_template(self):
        # Create a descriptive template dictionary
        template_mapping = {
            "Predicted Class Name 1": "Annotation Class Name 1",
            "Predicted Class Name 2": "Annotation Class Name 2",
            "Predicted Class Name 3": "Annotation Class Name 3",
            "Predicted Class Name 4": "Annotation Class Name 4",
            "Predicted Class Name 5": "Annotation Class Name 5",
            "Predicted Class Name 6": "Annotation Class Name 6",
            "Predicted Class Name 7": "Annotation Class Name 7",
            "Predicted Class Name 8": "Annotation Class Name 8",
            "Predicted Class Name 9": "Annotation Class Name 9",
            "Predicted Class Name 10": "Annotation Class Name 10",
            "Gibbons": "CrestedGibbons",
        }

        # Open a file dialog to select the save location
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Class Mapping Template", "class_mapping.json", "JSON Files (*.json)",
                                                   options=options)
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    json.dump(template_mapping, f, indent=4)
                self.results_text.setText(f"Class mapping template saved to {file_name}")
            except Exception as e:
                self.results_text.setText(f"Error saving class mapping template: {e}")

    def reset_results(self):
        self.results_text.clear()
        self.figure.clear()
        self.canvas.draw()
        #self.metrics_calculated = False  # Reset the flag

    def closeEvent(self, event):
        plt.close('all')
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set the dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    # Optionally, set a modern font
    font = QFont('Segoe UI', 10)
    app.setFont(font)

    # Apply a global stylesheet for QPushButton and QToolButton

    app.setStyleSheet("""
        QPushButton:hover {
            background-color: #294C7D;  /* Lighter blue when hovered */
        }
        QToolButton:hover {
            background-color: #294C7D;  /* Lighter blue when hovered */
        }
    """)

    window = PerformanceApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
