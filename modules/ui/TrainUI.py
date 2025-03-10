# train_ui.py

import sys
import json
import threading
import traceback
import webbrowser
from pathlib import Path
from collections.abc import Callable

import torch

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QFileDialog,
    QFrame,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QScrollArea,
    QProgressBar,
    QMessageBox
)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QCloseEvent


from modules.trainer.CloudTrainer import CloudTrainer
from modules.trainer.GenericTrainer import GenericTrainer
from modules.ui.AdditionalEmbeddingsTab import AdditionalEmbeddingsTab
from modules.ui.CaptionUI import CaptionUI
from modules.ui.CloudTab import CloudTab
from modules.ui.ConceptTab import ConceptTab
from modules.ui.ConvertModelUI import ConvertModelUI
from modules.ui.GeneralTab import GeneralTab
from modules.ui.LoraTab import LoraTab
from modules.ui.ModelTab import ModelTab
from modules.ui.ProfilingWindow import ProfilingWindow
from modules.ui.SampleWindow import SampleWindow
from modules.ui.SamplingTab import SamplingTab
from modules.ui.TopBar import TopBar
from modules.ui.TrainingTab import TrainingTab

from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ui.UIState import UIState
from modules.util.ui import components
from modules.zluda import ZLUDA


class TrainUI(QMainWindow):
    # For type hints
    set_step_progress: Callable[[int, int], None]
    set_epoch_progress: Callable[[int, int], None]
    tabview: QTabWidget

    def __init__(self):
        super().__init__()

        # -------------------------------------------------------------------
        # Basic window config
        # -------------------------------------------------------------------
        self.setWindowTitle("OneTrainer")
        self.resize(1100, 740)
        # If you want a fixed size:
        # self.setFixedSize(1100, 740)

        # In Qt, there's no built-in "appearance mode" setting like in customtkinter.
        # If you want styling, you'd typically apply style sheets or QPalette.

        # -------------------------------------------------------------------
        # Data / State
        # -------------------------------------------------------------------
        self.train_config = TrainConfig.default_values()
        self.ui_state = UIState(self, self.train_config)

        self.status_label = None
        self.training_button = None
        self.export_button = None
        self.tabview = None

        self.model_tab = None
        self.training_tab = None
        self.lora_tab = None
        self.cloud_tab = None
        self.additional_embeddings_tab = None

        self.training_thread = None
        self.training_callbacks = None
        self.training_commands = None

        # Persistent profiling window
        self.profiling_window = ProfilingWindow(self)

        # We'll store references for progress bars:
        self._step_progress_bar = None
        self._epoch_progress_bar = None

        # -------------------------------------------------------------------
        # Main Layout
        # -------------------------------------------------------------------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1) Top bar
        self.top_bar_component = self.create_top_bar()
        main_layout.addWidget(self.top_bar_component)

        # 2) Middle content
        content_frame = self.create_content_frame()
        main_layout.addWidget(content_frame, stretch=1)

        # 3) Bottom bar
        bottom_bar_frame = self.create_bottom_bar()
        main_layout.addWidget(bottom_bar_frame)

        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # -----------------------------------------------------------------------
    # Window Closing
    # -----------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent):
        """
        Called when user closes the window.
        """
        self.__close()
        event.accept()

    def __close(self):
        """
        Replaces your old __close() method.
        """
        self.top_bar_component.save_default()
        # If you'd like to exit the Qt application entirely:
        # self.close()  # not strictly needed if event.accept() was used

    # -----------------------------------------------------------------------
    # Changing model/training
    # -----------------------------------------------------------------------

    def change_model_type(self, model_type: ModelType):
        if self.model_tab:
            self.model_tab.refresh_ui()
        if self.training_tab:
            self.training_tab.refresh_ui()
        if self.lora_tab:
            self.lora_tab.refresh_ui()

    def change_training_method(self, training_method: TrainingMethod):
        if not self.tabview:
            return

        if self.model_tab:
            self.model_tab.refresh_ui()

        # remove "LoRA" tab if it exists
        if training_method != TrainingMethod.LORA and "LoRA" in self._tab_names():
            index = self.tabview.indexOf(self.lora_tab)
            if index >= 0:
                self.tabview.removeTab(index)
                self.lora_tab = None

        # remove "embedding" tab if it exists
        if training_method != TrainingMethod.EMBEDDING and "embedding" in self._tab_names():
            # find it, remove it
            # code depends on how you track "embedding_tab"
            idx = self.tabview.indexOfByName("embedding")  # not standard, you'd store references
            if idx >= 0:
                self.tabview.removeTab(idx)

        # add Lora tab if needed
        if training_method == TrainingMethod.LORA and "LoRA" not in self._tab_names():
            lora_widget = QWidget()
            self.lora_tab = LoraTab(lora_widget, self.train_config, self.ui_state)
            self.tabview.addTab(lora_widget, "LoRA")

        # add embedding tab if needed
        if training_method == TrainingMethod.EMBEDDING and "embedding" not in self._tab_names():
            embedding_widget = QWidget()
            self.embedding_tab(embedding_widget)
            self.tabview.addTab(embedding_widget, "embedding")

    def load_preset(self):
        # For your additional embeddings tab refresh, etc.
        if self.additional_embeddings_tab:
            self.additional_embeddings_tab.refresh_ui()

    # -----------------------------------------------------------------------
    # Top bar
    # -----------------------------------------------------------------------
    def create_top_bar(self) -> TopBar:
        top_bar_widget = TopBar(
            master=self,
            train_config=self.train_config,
            ui_state=self.ui_state,
            change_model_type_callback=self.change_model_type,
            change_training_method_callback=self.change_training_method,
            load_preset_callback=self.load_preset
        )
        return top_bar_widget

    # -----------------------------------------------------------------------
    # Bottom bar.
    # Contains progress bars, status label,
    # and buttons for training start, tensorboard popup, and export.
    # -----------------------------------------------------------------------
    def create_bottom_bar(self) -> QWidget:

        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        self._step_progress_bar = QProgressBar()
        self._epoch_progress_bar = QProgressBar()
        self._step_progress_bar.setValue(0)
        self._epoch_progress_bar.setValue(0)
        self._step_progress_bar.setFormat("step: %v / %m")
        self._epoch_progress_bar.setFormat("epoch: %v / %m")
        self._step_progress_bar.setTextVisible(True)
        self._epoch_progress_bar.setTextVisible(True)

        def _set_step_progress(value, max_value):
            self._step_progress_bar.setRange(0, max_value)
            self._step_progress_bar.setValue(value)

        def _set_epoch_progress(value, max_value):
            self._epoch_progress_bar.setRange(0, max_value)
            self._epoch_progress_bar.setValue(value)

        self.set_step_progress = _set_step_progress
        self.set_epoch_progress = _set_epoch_progress

        layout.addWidget(self._step_progress_bar)
        layout.addWidget(self._epoch_progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # spacer
        layout.addStretch(1)

        tensorboard_button = QPushButton("Tensorboard")
        tensorboard_button.clicked.connect(self.open_tensorboard)
        layout.addWidget(tensorboard_button)

        self.training_button = QPushButton("Start Training")
        self.training_button.clicked.connect(self.start_training)
        layout.addWidget(self.training_button)

        self.export_button = QPushButton("Export")
        self.export_button.setToolTip("Export the current configuration as a script to run without a UI")
        self.export_button.clicked.connect(self.export_training)
        layout.addWidget(self.export_button)

        return frame

    # -----------------------------------------------------------------------
    # Middle content: a tab widget, with tabs for each main function.
    # eg: "General", "Model", "Data", "Concepts", "Training", etc.
    # -----------------------------------------------------------------------
    def create_content_frame(self) -> QWidget:

        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        self.tabview = QTabWidget()
        layout.addWidget(self.tabview, stretch=1)

        # Create each tab
        general_tab = self.create_general_tab()
        self.tabview.addTab(general_tab, "general")

        self.model_tab = self.create_model_tab()
        self.tabview.addTab(self.model_tab, "model")

        data_tab = self.create_data_tab()
        self.tabview.addTab(data_tab, "data")

        concepts_tab = self.create_concepts_tab()
        self.tabview.addTab(concepts_tab, "concepts")

        self.training_tab = self.create_training_tab()
        self.tabview.addTab(self.training_tab, "training")

        sampling_tab = self.create_sampling_tab()
        self.tabview.addTab(sampling_tab, "sampling")

        backup_tab = self.create_backup_tab()
        self.tabview.addTab(backup_tab, "backup")

        tools_tab = self.create_tools_tab()
        self.tabview.addTab(tools_tab, "tools")

        self.additional_embeddings_tab = self.create_additional_embeddings_tab()
        self.tabview.addTab(self.additional_embeddings_tab, "additional embeddings")

        self.cloud_tab = self.create_cloud_tab()
        self.tabview.addTab(self.cloud_tab, "cloud")

        # initially set the training method
        self.change_training_method(self.train_config.training_method)

        return frame

    # -----------------------------------------------------------------------
    # Tab creation functions start here
    # -----------------------------------------------------------------------
    def create_general_tab(self) -> QWidget:

        return GeneralTab(self.ui_state)


    def create_model_tab(self, parent=None) -> QWidget:
        # In the original code, this is ModelTab(...)
        # We'll assume you've converted ModelTab to a QWidget-based class:
        return ModelTab(self, self.train_config, self.ui_state)

    def create_data_tab(self) -> QWidget:
        """
        PySide6 version of your create_data_tab function.
        Returns a QScrollArea with a container that has the relevant
        label/switch UI elements for data tab.
        """

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)

        scroll_area.setWidget(container)

        # row=0 => "Aspect Ratio Bucketing"
        components.label(
            container, 0, 0,
            "Aspect Ratio Bucketing",
            tooltip="Aspect ratio bucketing enables training on images with different aspect ratios"
        )
        components.switch(container, 0, 1, self.ui_state, "aspect_ratio_bucketing")

        # row=1 => "Latent Caching"
        components.label(
            container, 1, 0,
            "Latent Caching",
            tooltip="Caching of intermediate training data that can be re-used between epochs"
        )
        components.switch(container, 1, 1, self.ui_state, "latent_caching")

        # row=2 => "Clear cache before training"
        components.label(
            container, 2, 0,
            "Clear cache before training",
            tooltip=(
                "Clears the cache directory before starting to train. "
                "Only disable this if you want to continue using the same cached data. "
                "Disabling this can lead to errors if other settings are changed during a restart"
            )
        )
        components.switch(container, 2, 1, self.ui_state, "clear_cache_before_training")

        return scroll_area



    def create_concepts_tab(self) -> QWidget:
        # In your code: ConceptTab(master, self.train_config, self.ui_state)
        return ConceptTab(self, self.train_config, self.ui_state)

    def create_training_tab(self) -> TrainingTab:
        return TrainingTab(self, self.train_config, self.ui_state)

    def create_sampling_tab(self) -> QWidget:
        # In your code, you had some complex structure with "sample after" controls,
        # plus a SamplingTab(...) in a sub-frame. We'll do a basic version:
        w = QWidget()
        layout = QVBoxLayout(w)

        # "Sample Now" button
        sample_now_button = QPushButton("sample now")
        sample_now_button.clicked.connect(self.sample_now)
        layout.addWidget(sample_now_button)

        # "manual sample" => open_sample_ui
        manual_sample_button = QPushButton("manual sample")
        manual_sample_button.clicked.connect(self.open_sample_ui)
        layout.addWidget(manual_sample_button)

        # Then add the actual sampling options in a sub-area
        sampling_tab_widget = SamplingTab(self, self.train_config, self.ui_state)
        layout.addWidget(sampling_tab_widget)

        layout.addStretch(1)
        return w

    def create_backup_tab(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        container = QWidget()
        layout = QGridLayout(container)

        # "Backup Now" button
        backup_now_button = QPushButton("backup now")
        backup_now_button.clicked.connect(self.backup_now)
        layout.addWidget(backup_now_button, 0, 0)

        # "save now" button
        save_now_button = QPushButton("save now")
        save_now_button.clicked.connect(self.save_now)
        layout.addWidget(save_now_button, 0, 1)

        container.setLayout(layout)
        w.setWidget(container)
        return w

    def create_tools_tab(self) -> QWidget:
        w = QScrollArea()
        w.setWidgetResizable(True)
        container = QWidget()
        layout = QGridLayout(container)

        # "Open Dataset Tool"
        label_dataset = QLabel("Dataset Tools")
        button_dataset = QPushButton("Open")
        button_dataset.clicked.connect(self.open_dataset_tool)
        layout.addWidget(label_dataset, 0, 0)
        layout.addWidget(button_dataset, 0, 1)

        # "Convert Model Tool"
        label_convert = QLabel("Convert Model Tools")
        button_convert = QPushButton("Open")
        button_convert.clicked.connect(self.open_convert_model_tool)
        layout.addWidget(label_convert, 1, 0)
        layout.addWidget(button_convert, 1, 1)

        # "Sampling Tool"
        label_sampling = QLabel("Sampling Tool")
        button_sampling = QPushButton("Open")
        button_sampling.clicked.connect(self.open_sampling_tool)
        layout.addWidget(label_sampling, 2, 0)
        layout.addWidget(button_sampling, 2, 1)

        # "Profiling Tool"
        label_profiling = QLabel("Profiling Tool")
        button_profiling = QPushButton("Open")
        button_profiling.clicked.connect(self.open_profiling_tool)
        layout.addWidget(label_profiling, 3, 0)
        layout.addWidget(button_profiling, 3, 1)

        container.setLayout(layout)
        w.setWidget(container)
        return w

    def create_additional_embeddings_tab(self) -> QWidget:
        # AdditionalEmbeddingsTab(...) is presumably your own QWidget-based class
        return AdditionalEmbeddingsTab(self, self.train_config, self.ui_state)

    def create_cloud_tab(self) -> QWidget:
        # CloudTab(...) is presumably your own QWidget-based class
        return CloudTab(self, self.train_config, self.ui_state, parent=self)


    def _tab_names(self):
        return [self.tabview.tabText(i) for i in range(self.tabview.count())]

    def embedding_tab(self, widget: QWidget):
        """
        The old code used:
          self.embedding_tab(self.tabview.add("embedding"))
        We'll replicate a minimal approach. 
        """
        # e.g. fill the widget with controls for embedding
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Embedding tab content."))
        layout.addStretch(1)


    # -----------------------------------------------------------------------
    # Tensorboard
    # -----------------------------------------------------------------------
    def open_tensorboard(self):
        port = str(self.train_config.tensorboard_port)
        webbrowser.open(f"http://localhost:{port}", new=0, autoraise=False)

    # -----------------------------------------------------------------------
    # Train progress
    # -----------------------------------------------------------------------
    def on_update_train_progress(self, train_progress: TrainProgress, max_sample: int, max_epoch: int):
        self.set_step_progress(train_progress.epoch_step, max_sample)
        self.set_epoch_progress(train_progress.epoch, max_epoch)

    def on_update_status(self, status: str):
        if self.status_label:
            self.status_label.setText(status)

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------
    def open_dataset_tool(self):
        # Callback to open the dataset tool, aka CaptionUI
        dialog = CaptionUI(self, None, False)
        dialog.show()  # modal
        # or dialog.show() for modeless

    def open_convert_model_tool(self):
        dialog = ConvertModelUI(self)
        dialog.exec()

    def open_sampling_tool(self):
        if not self.training_callbacks and not self.training_commands:
            dialog = SampleWindow(self, train_config=self.train_config)
            dialog.exec()
            torch_gc()

    def open_profiling_tool(self):
        # In your ctk code: self.profiling_window.deiconify()
        # In Qt, just show/raise the window:
        self.profiling_window.show()
        self.profiling_window.raise_()

    def open_sample_ui(self):
        training_callbacks = self.training_callbacks
        training_commands = self.training_commands
        if training_callbacks and training_commands:
            dialog = SampleWindow(self, callbacks=training_callbacks, commands=training_commands)
            dialog.exec()
            training_callbacks.set_on_sample_custom()

    # -----------------------------------------------------------------------
    # Training Thread
    # -----------------------------------------------------------------------
    def __training_thread_function(self):
        error_caught = False

        self.training_callbacks = TrainCallbacks(
            on_update_train_progress=self.on_update_train_progress,
            on_update_status=self.on_update_status,
        )

        if self.train_config.cloud.enabled:
            trainer = CloudTrainer(
                self.train_config,
                self.training_callbacks,
                self.training_commands,
                reattach=self.cloud_tab.reattach
            )
        else:
            ZLUDA.initialize_devices(self.train_config)
            trainer = GenericTrainer(self.train_config, self.training_callbacks, self.training_commands)

        try:
            trainer.start()
            if self.train_config.cloud.enabled:
                self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)
            trainer.train()
        except Exception:
            if self.train_config.cloud.enabled:
                self.ui_state.get_var("secrets.cloud").update(self.train_config.secrets.cloud)
            error_caught = True
            traceback.print_exc()

        trainer.end()

        # clear gpu memory
        del trainer

        self.training_thread = None
        self.training_commands = None
        torch.clear_autocast_cache()
        torch_gc()

        if error_caught:
            self.on_update_status("error: check the console for more information")
        else:
            self.on_update_status("stopped")

        if self.training_button:
            self.training_button.setText("Start Training")
            self.training_button.setEnabled(True)

    def start_training(self):
        if self.training_thread is None:
            self.top_bar_component.save_default()

            if self.training_button:
                self.training_button.setText("Stop Training")
                self.training_button.setEnabled(True)

            self.training_commands = TrainCommands()

            self.training_thread = threading.Thread(target=self.__training_thread_function)
            self.training_thread.start()
        else:
            # i.e. stop training
            if self.training_button:
                self.training_button.setEnabled(False)
            self.on_update_status("stopping")
            if self.training_commands:
                self.training_commands.stop()

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    def export_training(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            ".",
            "JSON files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.train_config.to_pack_dict(secrets=False), f, indent=4)
            except Exception:
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # Training commands
    # -----------------------------------------------------------------------
    def sample_now(self):
        if self.training_commands:
            self.training_commands.sample_default()

    def backup_now(self):
        if self.training_commands:
            self.training_commands.backup()

    def save_now(self):
        if self.training_commands:
            self.training_commands.save()

