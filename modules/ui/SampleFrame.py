
# Might be better called "Sampling Frame"
# Contains one definition to take a sample, during training.
# (as handled under SamplingTab)

import os
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt

from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.ui.UIState import UIState
from modules.util.ui import components



class SampleFrame(QFrame):

    def __init__(
        self,
        parent,
        sample: SampleConfig,
        ui_state: UIState,
        include_prompt: bool = True,
        include_settings: bool = True,
    ):
        super().__init__()

        self.sample = sample
        self.ui_state = ui_state
        self.include_prompt = include_prompt
        self.include_settings = include_settings

        # Top-level layout for this frame
        self.layout_main = QGridLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(5)
        self.setLayout(self.layout_main)

        row_index = 0

        # If we want a "top_frame" for prompt
        if self.include_prompt:
            self.top_frame = QFrame(self)
            self.top_frame_layout = QGridLayout(self.top_frame)
            self.top_frame_layout.setContentsMargins(0, 0, 0, 0)
            self.top_frame_layout.setSpacing(5)
            self.top_frame.setLayout(self.top_frame_layout)

            # Place top_frame in row=0
            self.layout_main.addWidget(self.top_frame, row_index, 0, 1, 1, alignment=Qt.AlignTop)
            row_index += 1

        # If we want a "bottom_frame" for settings
        if self.include_settings:
            self.bottom_frame = QFrame(self)
            self.bottom_frame_layout = QGridLayout(self.bottom_frame)
            self.bottom_frame_layout.setContentsMargins(0, 0, 0, 0)
            self.bottom_frame_layout.setSpacing(5)
            self.bottom_frame.setLayout(self.bottom_frame_layout)

            # Place bottom_frame in row=1
            self.layout_main.addWidget(self.bottom_frame, row_index, 0, 1, 1, alignment=Qt.AlignTop)
            row_index += 1

        top_frame = self.top_frame if self.include_prompt else None
        bottom_frame = self.bottom_frame if self.include_settings else None

        if include_prompt:
            # prompt
            components.label(top_frame, 0, 0, "prompt:")
            components.entry(top_frame, 0, 1, self.ui_state, "prompt", width=600)

            # negative prompt
            components.label(top_frame, 1, 0, "negative prompt:")
            components.entry(top_frame, 1, 1, self.ui_state, "negative_prompt", width=600)

        if include_settings:
            # width
            components.label(bottom_frame, 0, 0, "width:")
            components.entry(bottom_frame, 0, 1, self.ui_state, "width")

            # height
            components.label(bottom_frame, 0, 2, "height:")
            components.entry(bottom_frame, 0, 3, self.ui_state, "height")

            # frames
            components.label(bottom_frame, 1, 0, "frames:",
                             tooltip="Number of frames to generate. Only used when generating videos.")
            components.entry(bottom_frame, 1, 1, self.ui_state, "frames")

            # length
            components.label(bottom_frame, 1, 2, "length:",
                             tooltip="Length in seconds of audio output.")
            components.entry(bottom_frame, 1, 3, self.ui_state, "length")

            # seed
            components.label(bottom_frame, 2, 0, "seed:")
            components.entry(bottom_frame, 2, 1, self.ui_state, "seed")

            # random seed
            components.label(bottom_frame, 2, 2, "random seed:")
            components.switch(bottom_frame, 2, 3, self.ui_state, "random_seed")

            # cfg scale
            components.label(bottom_frame, 3, 0, "cfg scale:")
            components.entry(bottom_frame, 3, 1, self.ui_state, "cfg_scale")

            # sampler
            components.label(bottom_frame, 4, 2, "sampler:")
            components.options_kv(bottom_frame, 4, 3, [
                ("DDIM", NoiseScheduler.DDIM),
                ("Euler", NoiseScheduler.EULER),
                ("Euler A", NoiseScheduler.EULER_A),
                # ("DPM++", NoiseScheduler.DPMPP), # TODO: produces noisy samples
                # ("DPM++ SDE", NoiseScheduler.DPMPP_SDE), # TODO: produces noisy samples
                ("UniPC", NoiseScheduler.UNIPC),
                ("Euler Karras", NoiseScheduler.EULER_KARRAS),
                ("DPM++ Karras", NoiseScheduler.DPMPP_KARRAS),
                ("DPM++ SDE Karras", NoiseScheduler.DPMPP_SDE_KARRAS),
                # ("UniPC Karras", NoiseScheduler.UNIPC_KARRAS),# TODO: update diffusers to fix UNIPC_KARRAS (see https://github.com/huggingface/diffusers/pull/4581)
            ], self.ui_state, "noise_scheduler")

            # steps
            components.label(bottom_frame, 4, 0, "steps:")
            components.entry(bottom_frame, 4, 1, self.ui_state, "diffusion_steps")

            # inpainting
            components.label(bottom_frame, 5, 0, "inpainting:",
                             tooltip="Enables inpainting sampling. Only available when sampling from an inpainting model.")
            components.switch(bottom_frame, 5, 1, self.ui_state, "sample_inpainting")

            # base image path
            components.label(bottom_frame, 6, 0, "base image path:",
                             tooltip="The base image used when inpainting.")
            components.file_entry(bottom_frame, 6, 1, self.ui_state, "base_image_path",
                                  allow_model_files=False,
                                  allow_image_files=True,
                                  )

            # mask image path
            components.label(bottom_frame, 7, 2, "mask image path:",
                             tooltip="The mask used when inpainting.")
            components.file_entry(bottom_frame, 7, 3, self.ui_state, "mask_image_path",
                                  allow_model_files=False,
                                  allow_image_files=True,
                                  )