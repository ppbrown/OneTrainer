# generate_captions_window.py

"""
Generates a pop-up window related to the CaptionsUI/data tools window,
specific to actually generating captions for a folder of images.

"""

import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QCheckBox, QProgressBar,
    QFileDialog, QGridLayout, QVBoxLayout
)
from PySide6.QtCore import Qt

# It looks like these are not used, but the imports trigger BaseImageCaptionModel
from modules.module.WDModel import WDModel
from modules.module.BlipModel import BlipModel
from modules.module.Blip2Model import Blip2Model
from modules.module.Moondream2Model import Moondream2Model

from modules.module.BaseImageCaptionModel import BaseImageCaptionModel

from modules.util.torch_util import default_device

import torch

class GenerateCaptionsWindow(QMainWindow):
    """
    Window for generating captions for a folder of images.
    """

        
    def __init__(self, parent, path, parent_include_subdirectories, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent  # reference to the parent, if you need to call parent methods
        self.setWindowTitle("Batch generate captions")
        self.resize(360, 360)  # or set a fixed size if you prefer: self.setFixedSize(360, 360)

        # Default path
        if path is None:
            path = ""

        # ---------------------------------------------------------------------
        # Variables / state
        # ---------------------------------------------------------------------
        self.caption_model_list = BaseImageCaptionModel.get_all_model_choices()
        self.caption_modelname_list = list(self.caption_model_list.keys())
        self.caption_modelname = self.caption_modelname_list[0]

        self.modes = ["Replace all captions", "Create if absent", "Add as new line"]

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # We’ll use a QGridLayout for row/column alignment
        grid = QGridLayout()
        layout.addLayout(grid)

        # Model label and combo
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.caption_modelname_list)
        # self.model_combo.setCurrentIndex(self.models.index("Blip")) 
        grid.addWidget(model_label, 0, 0)
        grid.addWidget(self.model_combo, 0, 1)

        # Path label, line edit, and browse button
        path_label = QLabel("Folder:")
        self.path_edit = QLineEdit(path)
        path_button = QPushButton("...")
        path_button.clicked.connect(self.browse_for_path)

        grid.addWidget(path_label, 1, 0)
        grid.addWidget(self.path_edit, 1, 1)
        grid.addWidget(path_button, 1, 2)

        # Initial caption
        caption_label = QLabel("Initial Caption:")
        self.caption_entry = QLineEdit()
        grid.addWidget(caption_label, 2, 0)
        grid.addWidget(self.caption_entry, 2, 1, 1, 2)

        # Caption prefix
        prefix_label = QLabel("Caption Prefix:")
        self.prefix_entry = QLineEdit()
        grid.addWidget(prefix_label, 3, 0)
        grid.addWidget(self.prefix_entry, 3, 1, 1, 2)

        # Caption postfix
        postfix_label = QLabel("Caption Postfix:")
        self.postfix_entry = QLineEdit()
        grid.addWidget(postfix_label, 4, 0)
        grid.addWidget(self.postfix_entry, 4, 1, 1, 2)

        # Mode label and combo
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.modes)
        self.mode_combo.setCurrentIndex(self.modes.index("Create if absent"))  # default
        grid.addWidget(mode_label, 5, 0)
        grid.addWidget(self.mode_combo, 5, 1, 1, 2)

        # Include subfolders
        subfolders_label = QLabel("Include subfolders:")
        self.include_sub_check = QCheckBox()
        self.include_sub_check.setChecked(bool(parent_include_subdirectories))
        grid.addWidget(subfolders_label, 6, 0)
        grid.addWidget(self.include_sub_check, 6, 1)

        # Progress label and bar
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_bar = QProgressBar()
        grid.addWidget(self.progress_label, 7, 0)
        grid.addWidget(self.progress_bar, 7, 1, 1, 2)

        # Create captions button
        create_button = QPushButton("Create Captions")
        create_button.clicked.connect(self.create_captions)
        grid.addWidget(create_button, 8, 0, 1, 3)

        # stretch to fill
        layout.addStretch(1)

        # Modal-like behavior (if you want)
        # self.setModal(True)  # If you want a truly modal dialog

    def browse_for_path(self):
        """
        Open a directory dialog, set the result to the path_edit line.
        """
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.path_edit.text())
        if chosen_dir:
            self.path_edit.setText(chosen_dir)

    def set_progress(self, value, max_value):
        """
        Update the progress bar and progress label.
        """
        if max_value == 0:
            percentage = 0
        else:
            percentage = int((value / max_value) * 100)

        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Progress: {value}/{max_value}")
        # If you want to ensure an immediate GUI refresh:
        # self.progress_bar.repaint()
        # self.progress_label.repaint()
        # QApplication.processEvents()

    def create_captions(self):
        """
        Called when "Create Captions" button is clicked. 
        """
        modelname = self.model_combo.currentText()

        if not modelname == self.caption_modelname:
            self.caption_model = self.caption_model_list[modelname](default_device, torch.float16, modelname)
        if self.caption_model:
            self.captionmodelname = modelname
        else:
            self.current_captionmodel = None    

        # Convert selected mode to your internal strings
        mode_map = {
            "Replace all captions": "replace",
            "Create if absent": "fill",
            "Add as new line": "add",
        }
        selected_mode = mode_map.get(self.mode_combo.currentText(), "fill")

        self.caption_model.caption_folder(
            sample_dir=self.path_edit.text(),
            initial_caption=self.caption_entry.text(),
            caption_prefix=self.prefix_entry.text(),
            caption_postfix=self.postfix_entry.text(),
            mode=selected_mode,
            progress_callback=self.set_progress,
            include_subdirectories=self.include_sub_check.isChecked(),
        )

        # Reload the parent’s image display or do any final updates
        self.parent.load_image

        # Optionally close the dialog automatically
        # self.accept()  # or self.close()
