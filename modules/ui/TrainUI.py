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
from modules.ui.ConceptsTab import ConceptsTab
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
        self.__close()
        event.accept()

    def __close(self):
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
    # --1---------------------------------------------------------------------
    def create_general_tab(self) -> QWidget:

        return GeneralTab(self.ui_state)


    def create_model_tab(self, parent=None) -> QWidget:
        # In the original code, this is ModelTab(...)
        # We'll assume you've converted ModelTab to a QWidget-based class:
        return ModelTab(self, self.train_config, self.ui_state)

    def create_data_tab(self) -> QWidget:

    
        scroll_area = QScrollArea()
        container = components.create_gridlayout(scroll_area)

        components.label(
            container, 0, 0,
            "Aspect Ratio Bucketing",
            tooltip="Aspect ratio bucketing enables training on images with different aspect ratios"
        )
        components.switch(container, 0, 1, self.ui_state, "aspect_ratio_bucketing")

        components.label(
            container, 1, 0,
            "Latent Caching",
            tooltip="Caching of intermediate training data that can be re-used between epochs"
        )
        components.switch(container, 1, 1, self.ui_state, "latent_caching")

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
        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)

        # legacy ugliness that creates the real contents behind the scenes
        self.conceptstab_configlist = ConceptsTab(container, self.train_config, self.ui_state)
        return container

    def create_training_tab(self) -> TrainingTab:
        return TrainingTab(self.train_config, self.ui_state)


    def create_sampling_tab(self) -> QWidget:

        # FIXLATER: All this extra widget setup should probably be moved
        # into SamplingTab
        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)

        # top_frame
        top_frame = QFrame(container)
        top_frame_layout = QGridLayout(top_frame)
        top_frame_layout.setContentsMargins(0,0,0,0)
        top_frame_layout.setSpacing(5)
        top_frame.setLayout(top_frame_layout)
        container_layout.addWidget(top_frame, 0, 0)

        # sub_frame
        sub_frame = QFrame(top_frame)
        sub_frame_layout = QGridLayout(sub_frame)
        sub_frame_layout.setContentsMargins(0,0,0,0)
        sub_frame_layout.setSpacing(5)
        sub_frame.setLayout(sub_frame_layout)
        top_frame_layout.addWidget(sub_frame, 1, 0, 1, 6)  # row=1 col=0..5

        # "Sample After" row=0 col=0..1
        components.label(top_frame, 0, 0, "Sample After",
                        tooltip="The interval used when automatically sampling from the model during training")
        components.time_entry(top_frame, 0, 1, self.ui_state, "sample_after", "sample_after_unit")

        # skip first
        components.label(top_frame, 0, 2, "Skip First",
                        tooltip="Start sampling automatically after this interval has elapsed.")
        components.entry(top_frame, 0, 3, self.ui_state, "sample_skip_first", width=50, sticky="nw")

        # format
        components.label(top_frame, 0, 4, "Format",
                        tooltip="File Format used when saving samples")
        components.options_kv(
            top_frame, 0, 5,
            [
                ("PNG", ImageFormat.PNG),
                ("JPG", ImageFormat.JPG),
            ],
            self.ui_state, "sample_image_format"
        )

        # sample now
        components.button(top_frame, 0, 6, "sample now", self.sample_now)
        # manual sample
        components.button(top_frame, 0, 7, "manual sample", self.open_sample_ui)

        # sub_frame row=0 col=0..3
        components.label(sub_frame, 0, 0, "Non-EMA Sampling",
                        tooltip="Whether to include non-ema sampling when using ema.")
        components.switch(sub_frame, 0, 1, self.ui_state, "non_ema_sampling")

        components.label(sub_frame, 0, 2, "Samples to Tensorboard",
                        tooltip="Whether to include sample images in the Tensorboard output.")
        components.switch(sub_frame, 0, 3, self.ui_state, "samples_to_tensorboard")

        # "frame" for table row=1 col=0
        """
        I think this is redundandt now??
        
        bottom_frame = QFrame(container)
        bottom_frame_layout = QGridLayout(bottom_frame)
        bottom_frame_layout.setContentsMargins(0,0,0,0)
        bottom_frame_layout.setSpacing(5)
        bottom_frame.setLayout(bottom_frame_layout)
        container_layout.addWidget(bottom_frame, 1, 0)

        # Have to save the object to avoid garbage collection for the internal callback
        self.samplingtab = SamplingTab(bottom_frame, self.train_config, self.ui_state)
        """
        bottom_frame = SamplingTab(container, self.train_config, self.ui_state)
        container_layout.addWidget(bottom_frame, 1, 0)

        return container


    def create_backup_tab(self) -> QWidget:
        scroll_area = QScrollArea()

        frame = components.create_gridlayout(scroll_area) 

        components.label(frame, 0, 0, "Backup After",
                         tooltip="The interval used when automatically creating model backups during training")
        components.time_entry(frame, 0, 1, self.ui_state, "backup_after", "backup_after_unit")

        components.button(frame, 0, 3, "backup now", self.backup_now)

        components.label(frame, 1, 0, "Rolling Backup",
                         tooltip="If rolling backups are enabled, older backups are deleted automatically")
        components.switch(frame, 1, 1, self.ui_state, "rolling_backup")

        components.label(frame, 1, 3, "Rolling Backup Count",
                         tooltip="Defines the number of backups to keep if rolling backups are enabled")
        components.entry(frame, 1, 4, self.ui_state, "rolling_backup_count")

        components.label(frame, 2, 0, "Backup Before Save",
                         tooltip="Create a full backup before saving the final model")
        components.switch(frame, 2, 1, self.ui_state, "backup_before_save")

        components.label(frame, 3, 0, "Save Every",
                         tooltip="The interval used when automatically saving the model during training")
        components.time_entry(frame, 3, 1, self.ui_state, "save_every", "save_every_unit")

        components.button(frame, 3, 3, "save now", self.save_now)

        components.label(frame, 4, 0, "Skip First",
                         tooltip="Start saving automatically after this interval has elapsed")
        components.entry(frame, 4, 1, self.ui_state, "save_skip_first", width=50, sticky="nw")

        components.label(frame, 5, 0, "Save Filename Prefix",
                         tooltip="The prefix for filenames used when saving the model during training")
        components.entry(frame, 5, 1, self.ui_state, "save_filename_prefix")



        return scroll_area


    def create_tools_tab(self) -> QWidget:
        # We use a widget-inside-a-widget, because QGridLayout is stupid and tries to take
        # over the whole scroll area.
        scroll_area = QScrollArea()
        grid_container = components.create_gridlayout(scroll_area)
        grid_layout = grid_container.layout()

        components.label(
            grid_container, 0, 0, "Dataset Tools",
            tooltip="Open the dataset tool for managing your training data"
        )
        components.button(
            grid_container, 0, 1, "Open",
            command=self.open_dataset_tool
        )
        components.label(
            grid_container, 1, 0, "Convert Model Tools",
            tooltip="Open the convert model tool for converting models to different formats"
        )
        components.button(
            grid_container, 1, 1, "Open",
            command=self.open_convert_model_tool
        )
        components.label(
            grid_container, 2, 0, "Sampling Tool",
            tooltip="Open the sampling tool for generating images from your model"
        )
        components.button(
            grid_container, 2, 1, "Open",
            command=self.open_sampling_tool
        )
        components.label(
            grid_container, 3, 0, "Profiling Tool",
            tooltip="Open the profiling tool for analyzing your model's performance"
        )
        components.button(
            grid_container, 3, 1, "Open",
            command=self.open_profiling_tool
        )

        return scroll_area

    def create_additional_embeddings_tab(self) -> QWidget:
        """
        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        """

        return AdditionalEmbeddingsTab(self, self.train_config, self.ui_state)


    def create_cloud_tab(self) -> QWidget:
        # CloudTab(...) is presumably your own QWidget-based class
        return CloudTab(self.train_config, self.ui_state, self)


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

