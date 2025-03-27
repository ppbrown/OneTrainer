
# Window to edit properties of a single "concept" from the ConceptTab

import os
import pathlib
import random
import traceback

import torch
from PIL import Image
from torchvision.transforms import functional

from PySide6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QScrollArea,
    QFrame, QPlainTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

from modules.util import path_util
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.BalancingStrategy import BalancingStrategy
from modules.util.ui import components
from modules.util.ui.UIState import UIState

# mgds pipeline modules
from mgds.LoadingPipeline import LoadingPipeline
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class InputPipelineModule(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def length(self) -> int:
        return 1

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return self.data


class ConceptWindow(QDialog):
    def __init__(
        self,
        parent,
        concept,
        ui_state: UIState,
        image_ui_state: UIState,
        text_ui_state: UIState,
        *args,
        **kwargs
    ):
        super().__init__()
        self.concept = concept
        self.ui_state = ui_state
        self.image_ui_state = image_ui_state
        self.text_ui_state = text_ui_state

        self.image_preview_file_index = 0

        # Setup QDialog
        self.setWindowTitle("Concept")
        self.resize(800, 700)
        # For a blocking dialog, you'd do self.setModal(True)

        # Main layout for the entire dialog
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)

        # QTabWidget in row=0
        self.tabview = QTabWidget(self)
        self.main_layout.addWidget(self.tabview, 0, 0, 1, 1)

        # Add tabs
        self.general_tab_widget = QWidget()
        self.tabview.addTab(self.general_tab_widget, "general")

        self.image_augmentation_widget = QWidget()
        self.tabview.addTab(self.image_augmentation_widget, "image augmentation")

        self.text_augmentation_widget = QWidget()
        self.tabview.addTab(self.text_augmentation_widget, "text augmentation")

        # Build each tab
        self.__general_tab(self.general_tab_widget)
        self.__image_augmentation_tab(self.image_augmentation_widget)
        self.__text_augmentation_tab(self.text_augmentation_widget)

        # "OK" button at row=1
        self.ok_button = QPushButton("ok", self)
        self.ok_button.clicked.connect(self.__ok)
        self.main_layout.addWidget(self.ok_button, 1, 0, 1, 1, alignment=Qt.AlignRight)

    def __general_tab(self, master: QWidget):
        """
        Replaces your ctk.CTkScrollableFrame with a QScrollArea,
        then place everything in a QGridLayout inside.
        """
        # QScrollArea
        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0,0,0,0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        # Container inside the scroll
        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5,5,5,5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        # name
        components.label(container, 0, 0, "Name", tooltip="Name of the concept")
        components.entry(container, 0, 1, self.ui_state, "name")

        # enabled
        components.label(container, 1, 0, "Enabled", tooltip="Enable or disable this concept")
        components.switch(container, 1, 1, self.ui_state, "enabled")

        # validation_concept
        components.label(container, 2, 0, "Validation concept", tooltip="Use concept for validation instead of training")
        components.switch(container, 2, 1, self.ui_state, "validation_concept")

        # path
        components.label(container, 3, 0, "Path", tooltip="Path for training data")
        components.dir_entry(container, 3, 1, self.ui_state, "path")

        # prompt source
        components.label(container, 4, 0, "Prompt Source",
                         tooltip="The source of prompts used. 'From text file per sample', etc.")
        prompt_path_entry = components.file_entry(container, 4, 2, self.text_ui_state, "prompt_path")

        def set_prompt_path_entry_enabled(option: str):
            # Enable the file entry's sub-widgets only if 'concept' is chosen.
            for child in prompt_path_entry.children():
                child.setEnabled(option == 'concept')


        components.options_kv(
            container, 4, 1,
            [
                ("From text file per sample", 'sample'),
                ("From single text file", 'concept'),
                ("From image file name", 'filename'),
            ],
            self.text_ui_state, "prompt_source", command=set_prompt_path_entry_enabled
        )
        set_prompt_path_entry_enabled(self.concept.text.prompt_source)

        # include subdirectories
        components.label(container, 5, 0, "Include Subdirectories")
        components.switch(container, 5, 1, self.ui_state, "include_subdirectories")

        # image variations
        components.label(container, 6, 0, "Image Variations")
        components.entry(container, 6, 1, self.ui_state, "image_variations")

        # text variations
        components.label(container, 7, 0, "Text Variations")
        components.entry(container, 7, 1, self.ui_state, "text_variations")

        # balancing
        components.label(container, 8, 0, "Balancing")
        components.entry(container, 8, 1, self.ui_state, "balancing")
        components.options(container, 8, 2, [str(x) for x in list(BalancingStrategy)], self.ui_state, "balancing_strategy")

        # loss weight
        components.label(container, 9, 0, "Loss Weight")
        components.entry(container, 9, 1, self.ui_state, "loss_weight")

    def __image_augmentation_tab(self, master: QWidget):
        """
        Window with with a grid for the 'Random' and 'Fixed' columns, plus an image preview.
        """
        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0,0,0,0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5,5,5,5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        # header
        components.label(container, 0, 1, "Random")
        components.label(container, 0, 2, "Fixed")

        # crop jitter
        components.label(container, 1, 0, "Crop Jitter")
        components.switch(container, 1, 1, self.image_ui_state, "enable_crop_jitter")

        # random flip
        components.label(container, 2, 0, "Random Flip")
        components.switch(container, 2, 1, self.image_ui_state, "enable_random_flip")
        components.switch(container, 2, 2, self.image_ui_state, "enable_fixed_flip")

        # random rotate
        components.label(container, 3, 0, "Random Rotation")
        components.switch(container, 3, 1, self.image_ui_state, "enable_random_rotate")
        components.switch(container, 3, 2, self.image_ui_state, "enable_fixed_rotate")
        components.entry(container, 3, 3, self.image_ui_state, "random_rotate_max_angle")

        # brightness
        components.label(container, 4, 0, "Random Brightness")
        components.switch(container, 4, 1, self.image_ui_state, "enable_random_brightness")
        components.switch(container, 4, 2, self.image_ui_state, "enable_fixed_brightness")
        components.entry(container, 4, 3, self.image_ui_state, "random_brightness_max_strength")

        # contrast
        components.label(container, 5, 0, "Random Contrast")
        components.switch(container, 5, 1, self.image_ui_state, "enable_random_contrast")
        components.switch(container, 5, 2, self.image_ui_state, "enable_fixed_contrast")
        components.entry(container, 5, 3, self.image_ui_state, "random_contrast_max_strength")

        # saturation
        components.label(container, 6, 0, "Random Saturation")
        components.switch(container, 6, 1, self.image_ui_state, "enable_random_saturation")
        components.switch(container, 6, 2, self.image_ui_state, "enable_fixed_saturation")
        components.entry(container, 6, 3, self.image_ui_state, "random_saturation_max_strength")

        # hue
        components.label(container, 7, 0, "Random Hue")
        components.switch(container, 7, 1, self.image_ui_state, "enable_random_hue")
        components.switch(container, 7, 2, self.image_ui_state, "enable_fixed_hue")
        components.entry(container, 7, 3, self.image_ui_state, "random_hue_max_strength")

        # circular mask shrink
        components.label(container, 8, 0, "Circular Mask Generation")
        components.switch(container, 8, 1, self.image_ui_state, "enable_random_circular_mask_shrink")

        # random rotate/crop
        components.label(container, 9, 0, "Random Rotate and Crop")
        components.switch(container, 9, 1, self.image_ui_state, "enable_random_mask_rotate_crop")

        # resolution override
        components.label(container, 10, 0, "Resolution Override")
        components.switch(container, 10, 2, self.image_ui_state, "enable_resolution_override")
        components.entry(container, 10, 3, self.image_ui_state, "resolution_override")

        # image preview 
        image_preview, filename_preview, caption_preview = self.__get_preview_image()
        self.preview_pixmap = self.__pil_to_qpixmap(image_preview)

        self.image_label = QLabel(container)
        self.image_label.setPixmap(self.preview_pixmap)
        self.image_label.setFixedSize(min(image_preview.width,300), min(image_preview.height,300))
        self.image_label.setScaledContents(True)
        container_layout.addWidget(self.image_label, 0, 4, 6, 1)  # row=0..5

        # refresh preview buttons
        update_button_frame = QFrame(container)
        update_button_layout = QGridLayout(update_button_frame)
        update_button_layout.setContentsMargins(0,0,0,0)
        update_button_layout.setSpacing(5)
        update_button_frame.setLayout(update_button_layout)
        container_layout.addWidget(update_button_frame, 6, 4, 1, 1)

        self.prev_preview_button = QPushButton("<", update_button_frame)
        self.prev_preview_button.setFixedWidth(40)
        self.prev_preview_button.clicked.connect(self.__prev_image_preview)
        update_button_layout.addWidget(self.prev_preview_button, 0, 0)

        self.update_preview_button = QPushButton("Update Preview", update_button_frame)
        self.update_preview_button.clicked.connect(self.__update_image_preview)
        update_button_layout.addWidget(self.update_preview_button, 0, 1)

        self.next_preview_button = QPushButton(">", update_button_frame)
        self.next_preview_button.setFixedWidth(40)
        self.next_preview_button.clicked.connect(self.__next_image_preview)
        update_button_layout.addWidget(self.next_preview_button, 0, 2)

        # filename preview
        self.filename_preview_label = QLabel(container)
        self.filename_preview_label.setText(filename_preview)
        self.filename_preview_label.setFixedWidth(300)
        container_layout.addWidget(self.filename_preview_label, 7, 4, 1, 1)

        # caption preview
        self.caption_preview = QPlainTextEdit(container)
        self.caption_preview.setFixedSize(300, 150)
        self.caption_preview.setReadOnly(True)
        self.caption_preview.insertPlainText(caption_preview)
        container_layout.addWidget(self.caption_preview, 8, 4, 4, 1)

    def __text_augmentation_tab(self, master: QWidget):
        """
        Another QScrollArea with a container that has a QGridLayout for text augmentation controls.
        """
        scroll_area = QScrollArea(master)
        scroll_area.setWidgetResizable(True)
        layout_master = QVBoxLayout(master)
        layout_master.setContentsMargins(0,0,0,0)
        layout_master.setSpacing(5)
        master.setLayout(layout_master)
        layout_master.addWidget(scroll_area)

        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5,5,5,5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        scroll_area.setWidget(container)

        # tag shuffling
        components.label(container, 0, 0, "Tag Shuffling")
        components.switch(container, 0, 1, self.text_ui_state, "enable_tag_shuffling")

        # tag delimiter
        components.label(container, 1, 0, "Tag Delimiter")
        components.entry(container, 1, 1, self.text_ui_state, "tag_delimiter")

        # keep_tags_count
        components.label(container, 2, 0, "Keep Tag Count")
        components.entry(container, 2, 1, self.text_ui_state, "keep_tags_count")

        # tag dropout
        components.label(container, 3, 0, "Tag Dropout")
        components.switch(container, 3, 1, self.text_ui_state, "tag_dropout_enable")

        # dropout mode
        components.label(container, 4, 0, "Dropout Mode")
        components.options_kv(
            container, 4, 1,
            [
                ("Full", 'FULL'),
                ("Random", 'RANDOM'),
                ("Random Weighted", 'RANDOM WEIGHTED'),
            ],
            self.text_ui_state, "tag_dropout_mode", None
        )
        components.label(container, 4, 2, "Probability")
        components.entry(container, 4, 3, self.text_ui_state, "tag_dropout_probability")

        # special dropout tags
        components.label(container, 5, 0, "Special Dropout Tags")
        components.options_kv(
            container, 5, 1,
            [
                ("None", 'NONE'),
                ("Blacklist", 'BLACKLIST'),
                ("Whitelist", 'WHITELIST'),
            ],
            self.text_ui_state, "tag_dropout_special_tags_mode", None
        )
        components.entry(container, 5, 2, self.text_ui_state, "tag_dropout_special_tags")

        # special tags regex
        components.label(container, 6, 0, "Special Tags Regex")
        components.switch(container, 6, 1, self.text_ui_state, "tag_dropout_special_tags_regex")

        # randomize capitalization
        components.label(container, 7, 0, "Randomize Capitalization")
        components.switch(container, 7, 1, self.text_ui_state, "caps_randomize_enable")
        components.label(container, 7, 2, "Force Lowercase")
        components.switch(container, 7, 3, self.text_ui_state, "caps_randomize_lowercase")

        # capitalization mode
        components.label(container, 8, 0, "Capitalization Mode")
        components.entry(container, 8, 1, self.text_ui_state, "caps_randomize_mode")
        components.label(container, 8, 2, "Probability")
        components.entry(container, 8, 3, self.text_ui_state, "caps_randomize_probability")

    def __prev_image_preview(self):
        self.image_preview_file_index = max(self.image_preview_file_index - 1, 0)
        self.__update_image_preview()

    def __next_image_preview(self):
        self.image_preview_file_index += 1
        self.__update_image_preview()

    def __update_image_preview(self):
        image_preview, filename_preview, caption_preview = self.__get_preview_image()
        self.preview_pixmap = self.__pil_to_qpixmap(image_preview)
        self.image_label.setPixmap(self.preview_pixmap)
        self.image_label.setFixedSize(min(image_preview.width,300), min(image_preview.height,300))
        self.image_label.setScaledContents(True)

        self.filename_preview_label.setText(filename_preview)
        self.caption_preview.setReadOnly(False)
        self.caption_preview.clear()
        self.caption_preview.insertPlainText(caption_preview)
        self.caption_preview.setReadOnly(True)

    def __get_preview_image(self):
        """
        pick an image from the concept.path, apply MGDS pipeline.
        """
        preview_image_path = "resources/icons/icon.png"
        file_index = -1
        glob_pattern = "**/*.*" if self.concept.include_subdirectories else "*.*"

        if os.path.isdir(self.concept.path):
            for path in pathlib.Path(self.concept.path).glob(glob_pattern):
                extension = os.path.splitext(path)[1]
                if (path.is_file()
                    and path_util.is_supported_image_extension(extension)
                    and not path.name.endswith("-masklabel.png")):
                    preview_image_path = path_util.canonical_join(self.concept.path, path)
                    file_index += 1
                    if file_index == self.image_preview_file_index:
                        break

        image = Image.open(preview_image_path).convert("RGB")
        image_tensor = functional.to_tensor(image)

        splitext = os.path.splitext(preview_image_path)
        preview_mask_path = path_util.canonical_join(splitext[0] + "-masklabel.png")
        if not os.path.isfile(preview_mask_path):
            preview_mask_path = None

        if preview_mask_path:
            mask = Image.open(preview_mask_path).convert("L")
            mask_tensor = functional.to_tensor(mask)
        else:
            mask_tensor = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]))

        # Prepare pipeline input data
        input_module = InputPipelineModule({
            'true': True,
            'image': image_tensor,
            'mask': mask_tensor,
            'enable_random_flip': self.concept.image.enable_random_flip,
            'enable_fixed_flip': self.concept.image.enable_fixed_flip,
            'enable_random_rotate': self.concept.image.enable_random_rotate,
            'enable_fixed_rotate': self.concept.image.enable_fixed_rotate,
            'random_rotate_max_angle': self.concept.image.random_rotate_max_angle,
            'enable_random_brightness': self.concept.image.enable_random_brightness,
            'enable_fixed_brightness': self.concept.image.enable_fixed_brightness,
            'random_brightness_max_strength': self.concept.image.random_brightness_max_strength,
            'enable_random_contrast': self.concept.image.enable_random_contrast,
            'enable_fixed_contrast': self.concept.image.enable_fixed_contrast,
            'random_contrast_max_strength': self.concept.image.random_contrast_max_strength,
            'enable_random_saturation': self.concept.image.enable_random_saturation,
            'enable_fixed_saturation': self.concept.image.enable_fixed_saturation,
            'random_saturation_max_strength': self.concept.image.random_saturation_max_strength,
            'enable_random_hue': self.concept.image.enable_random_hue,
            'enable_fixed_hue': self.concept.image.enable_fixed_hue,
            'random_hue_max_strength': self.concept.image.random_hue_max_strength,
            'enable_random_circular_mask_shrink': self.concept.image.enable_random_circular_mask_shrink,
            'enable_random_mask_rotate_crop': self.concept.image.enable_random_mask_rotate_crop,
        })

        # MGDS modules
        circular_mask_shrink = RandomCircularMaskShrink(
            mask_name='mask', shrink_probability=1.0,
            shrink_factor_min=0.2, shrink_factor_max=1.0,
            enabled_in_name='enable_random_circular_mask_shrink'
        )
        random_mask_rotate_crop = RandomMaskRotateCrop(
            mask_name='mask',
            additional_names=['image'],
            min_size=512,
            min_padding_percent=10,
            max_padding_percent=30,
            max_rotate_angle=20,
            enabled_in_name='enable_random_mask_rotate_crop'
        )
        random_flip = RandomFlip(
            names=['image','mask'],
            enabled_in_name='enable_random_flip',
            fixed_enabled_in_name='enable_fixed_flip'
        )
        random_rotate = RandomRotate(
            names=['image','mask'],
            enabled_in_name='enable_random_rotate',
            fixed_enabled_in_name='enable_fixed_rotate',
            max_angle_in_name='random_rotate_max_angle'
        )
        random_brightness = RandomBrightness(
            names=['image'],
            enabled_in_name='enable_random_brightness',
            fixed_enabled_in_name='enable_fixed_brightness',
            max_strength_in_name='random_brightness_max_strength'
        )
        random_contrast = RandomContrast(
            names=['image'],
            enabled_in_name='enable_random_contrast',
            fixed_enabled_in_name='enable_fixed_contrast',
            max_strength_in_name='random_contrast_max_strength'
        )
        random_saturation = RandomSaturation(
            names=['image'],
            enabled_in_name='enable_random_saturation',
            fixed_enabled_in_name='enable_fixed_saturation',
            max_strength_in_name='random_saturation_max_strength'
        )
        random_hue = RandomHue(
            names=['image'],
            enabled_in_name='enable_random_hue',
            fixed_enabled_in_name='enable_fixed_hue',
            max_strength_in_name='random_hue_max_strength'
        )
        output_module = OutputPipelineModule(['image','mask'])

        modules = [
            input_module,
            circular_mask_shrink,
            random_mask_rotate_crop,
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            output_module,
        ]

        from mgds.LoadingPipeline import LoadingPipeline
        pipeline = LoadingPipeline(
            device=torch.device('cpu'),
            modules=modules,
            batch_size=1,
            seed=random.randint(0, 2**30),
            state=None,
            initial_epoch=0,
            initial_index=0
        )

        data = pipeline.__next__()
        image_tensor = data['image']
        mask_tensor = data['mask']

        # filename + first line of base caption
        filename_output = os.path.basename(preview_image_path)
        try:
            if self.concept.text.prompt_source == "sample":
                with open(splitext[0] + ".txt", "r", encoding="utf-8") as prompt_file:
                    prompt_output = prompt_file.readline()
            elif self.concept.text.prompt_source == "filename":
                prompt_output = os.path.splitext(os.path.basename(preview_image_path))[0]
            elif self.concept.text.prompt_source == "concept":
                with open(self.concept.text.prompt_path, "r", encoding="utf-8") as prompt_file:
                    prompt_output = prompt_file.readline()
            else:
                prompt_output = "No caption found."
        except FileNotFoundError:
            prompt_output = "No caption found."

        mask_tensor = torch.clamp(mask_tensor, 0.3, 1)
        image_tensor = image_tensor * mask_tensor
        out_image = functional.to_pil_image(image_tensor)
        out_image.thumbnail((300, 300))

        return out_image, filename_output, prompt_output

    def __pil_to_qpixmap(self, pil_image: Image.Image) -> QPixmap:
        """
        Convert a PIL Image to a QPixmap for display in a QLabel.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(data, pil_image.width, pil_image.height, 3*pil_image.width, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def __ok(self):
        # self.concept.configure_element()
        self.accept()  # or self.accept() if you want to close the dialog
