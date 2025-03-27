
# This handles a MANUAL sample of a current model.
# Automated sampling is handled by SampleTab and SampleFrame

import copy
import os
import torch
from PIL import Image

from PySide6.QtWidgets import (
    QDialog, QLabel, QPushButton, QGridLayout, QProgressBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

from modules.model.BaseModel import BaseModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.ui.SampleFrame import SampleFrame 
from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.time_util import get_string_timestamp
from modules.util.ui.UIState import UIState


class SampleWindow(QDialog):


    def __init__(
        self,
        parent=None,
        train_config: TrainConfig | None = None,
        callbacks: TrainCallbacks | None = None,
        commands: TrainCommands | None = None,
        *args,
        **kwargs
    ):
        super().__init__(parent, *args, **kwargs)

        self.current_train_config = train_config
        self.callbacks = callbacks
        self.commands = commands

        # If we have a local train_config, make a copy for sampling:
        if train_config is not None:
            self.initial_train_config = TrainConfig.default_values().from_dict(
                train_config.to_dict()
            )
            # remove some settings to speed up model loading
            self.initial_train_config.optimizer.optimizer = None
            self.initial_train_config.ema = EMAMode.OFF
        else:
            self.initial_train_config = None

        # If no local config => rely on external (commands/callbacks)
        use_external_model = (self.initial_train_config is None)

        if use_external_model and self.callbacks:
            # external sampling usage
            self.callbacks.set_on_sample_custom(self.__update_preview)
            self.callbacks.set_on_update_sample_custom_progress(self.__update_progress)
            self.model = None
            self.model_sampler = None
        else:
            self.model = None
            self.model_sampler = None

        # Our sample config + UI state
        self.sample = SampleConfig.default_values()
        self.ui_state = UIState(self, self.sample)

        self.setWindowTitle("Sample")
        self.resize(1200, 800)
        self.setModal(False)  # modeless; or True if you want a blocking dialog

        # Layout
        self.layout_grid = QGridLayout(self)
        self.layout_grid.setContentsMargins(5, 5, 5, 5)
        self.layout_grid.setSpacing(5)
        self.setLayout(self.layout_grid)

        # Row 0: prompt frame
        self.prompt_frame = SampleFrame(
            parent=self,
            sample=self.sample,
            ui_state=self.ui_state,
            include_prompt=True,
            include_settings=False
        )
        self.layout_grid.addWidget(self.prompt_frame, 0, 0, 1, 2)

        # Row 1: settings frame at col=0
        self.settings_frame = SampleFrame(
            parent=self,
            sample=self.sample,
            ui_state=self.ui_state,
            include_prompt=False,
            include_settings=True
        )
        self.layout_grid.addWidget(self.settings_frame, 1, 0)

        # Row 1 col=1 => image label (spanning rows=1..3)
        self.image_label = QLabel()
        self.image_label.setFixedSize(512, 512)
        self.layout_grid.addWidget(self.image_label, 1, 1, 3, 1)

        # Initialize with a dummy black image
        self.__set_image(self.__dummy_image())

        # Row 2 col=0 => progress bar
        self.progress_bar = QProgressBar()
        self.layout_grid.addWidget(self.progress_bar, 2, 0)

        # Row 3 col=0 => "sample" button
        self.sample_button = QPushButton("sample")
        self.sample_button.clicked.connect(self.__sample)
        self.layout_grid.addWidget(self.sample_button, 3, 0)

    def __load_model(self) -> BaseModel:
        """
        Load the model from self.initial_train_config for local sampling.
        """
        model_loader = create.create_model_loader(
            model_type=self.initial_train_config.model_type,
            training_method=self.initial_train_config.training_method
        )
        model_setup = create.create_model_setup(
            model_type=self.initial_train_config.model_type,
            train_device=torch.device(self.initial_train_config.train_device),
            temp_device=torch.device(self.initial_train_config.temp_device),
            training_method=self.initial_train_config.training_method
        )

        model_names = self.initial_train_config.model_names()

        if self.initial_train_config.continue_last_backup:
            last_backup_path = self.initial_train_config.get_last_backup_path()
            if last_backup_path:
                if self.initial_train_config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.initial_train_config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:
                    model_names.base_model = last_backup_path
                print(f"Loading from backup '{last_backup_path}'...")
            else:
                print("No backup found, loading without backup...")

        model = model_loader.load(
            model_type=self.initial_train_config.model_type,
            model_names=model_names,
            weight_dtypes=self.initial_train_config.weight_dtypes()
        )
        model.train_config = self.initial_train_config

        model_setup.setup_optimizations(model, self.initial_train_config)
        model_setup.setup_train_device(model, self.initial_train_config)
        model_setup.setup_model(model, self.initial_train_config)
        model.to(torch.device(self.initial_train_config.temp_device))

        return model

    def __create_sampler(self, model: BaseModel) -> BaseModelSampler:
        """
        Create a BaseModelSampler from the given model.
        """
        return create.create_model_sampler(
            train_device=torch.device(self.initial_train_config.train_device),
            temp_device=torch.device(self.initial_train_config.temp_device),
            model=model,
            model_type=self.initial_train_config.model_type,
            training_method=self.initial_train_config.training_method
        )

    def __update_preview(self, sampler_output: ModelSamplerOutput):
        """
        Called when a sample is produced externally or locally. If it's an IMAGE, show it.
        """
        if sampler_output.file_type == FileType.IMAGE:
            pil_image = sampler_output.data
            self.__set_image(pil_image)

    def __update_progress(self, progress: int, max_progress: int):
        """
        Update the progress bar. This is called by external or local sampling code.
        """
        self.progress_bar.setRange(0, max_progress)
        self.progress_bar.setValue(progress)
        self.update()

    def __dummy_image(self) -> Image.Image:
        """
        Return a 512x512 black image.
        """
        return Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))

    def __set_image(self, pil_image: Image.Image):
        """
        Convert a PIL Image to QPixmap and set it on the image_label.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(data, pil_image.width, pil_image.height, 3 * pil_image.width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def __sample(self):
        """
        Called by the "sample" button. Either delegates to commands or does local sampling.
        """
        sample_cfg = copy.copy(self.sample)

        if self.commands:
            # Let the external commands handle it
            self.commands.sample_custom(sample_cfg)
        else:
            # Local sampling
            if self.model is None:
                # lazy initialization
                self.model = self.__load_model()
                self.model_sampler = self.__create_sampler(self.model)

            sample_cfg.from_train_config(self.current_train_config)

            sample_dir = os.path.join(
                self.initial_train_config.workspace_dir,
                "samples",
                "custom"
            )
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir, exist_ok=True)

            progress = self.model.train_progress
            if progress:
                progress_str = progress.filename_string()
            else:
                progress_str = "no_progress"

            dest_filename = f"{get_string_timestamp()}-training-sample-{progress_str}"
            sample_path = os.path.join(sample_dir, dest_filename)

            self.model.eval()

            self.model_sampler.sample(
                sample_config=sample_cfg,
                destination=sample_path,
                image_format=self.current_train_config.sample_image_format,
                video_format=self.current_train_config.sample_video_format,
                audio_format=self.current_train_config.sample_audio_format,
                on_sample=self.__update_preview,
                on_update_progress=self.__update_progress
            )
