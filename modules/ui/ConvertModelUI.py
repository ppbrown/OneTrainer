
# Util window under the "tools" tab

import traceback
import os
import uuid

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog
)
from PySide6.QtCore import Qt

from modules.util import create
from modules.util.args.ConvertModelArgs import ConvertModelArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModelNames import EmbeddingName, ModelNames
from modules.util.torch_util import torch_gc
from modules.util.ui.UIState import UIState


class ConvertModelUI(QDialog):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.setWindowTitle("Convert models")
        self.resize(550, 350)
        # self.setModal(False)  # if you want a non-blocking dialog

        # Data / State
        self.convert_model_args = ConvertModelArgs.default_values()
        self.ui_state = UIState(self, self.convert_model_args)

        # Main layout
        self.layout_main = QGridLayout(self)
        self.layout_main.setContentsMargins(10, 10, 10, 10)
        self.layout_main.setSpacing(10)
        self.setLayout(self.layout_main)

        # Build the UI
        self.__create_ui()

    def __create_ui(self):
        """
        Replaces your main_frame(...) method. We'll place the same fields in a QGridLayout.
        """

        # 0) Model Type
        lbl_model_type = QLabel("Model Type")
        lbl_model_type.setToolTip("Type of the model")
        self.layout_main.addWidget(lbl_model_type, 0, 0)

        self.combo_model_type = QComboBox()
        # Add items
        self.combo_model_type.addItem("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15)
        self.combo_model_type.addItem("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING)
        self.combo_model_type.addItem("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20)
        self.combo_model_type.addItem("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING)
        self.combo_model_type.addItem("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21)
        self.combo_model_type.addItem("Stable Diffusion 3", ModelType.STABLE_DIFFUSION_3)
        self.combo_model_type.addItem("Stable Diffusion 3.5", ModelType.STABLE_DIFFUSION_35)
        self.combo_model_type.addItem("Stable Diffusion XL 1.0 Base", ModelType.STABLE_DIFFUSION_XL_10_BASE)
        self.combo_model_type.addItem("Stable Diffusion XL 1.0 Base Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING)
        self.combo_model_type.addItem("Wuerstchen v2", ModelType.WUERSTCHEN_2)
        self.combo_model_type.addItem("Stable Cascade", ModelType.STABLE_CASCADE_1)
        self.combo_model_type.addItem("PixArt Alpha", ModelType.PIXART_ALPHA)
        self.combo_model_type.addItem("PixArt Sigma", ModelType.PIXART_SIGMA)
        self.combo_model_type.addItem("Flux Dev", ModelType.FLUX_DEV_1)
        self.combo_model_type.addItem("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1)
        self.combo_model_type.addItem("Hunyuan Video", ModelType.HUNYUAN_VIDEO)
        self.layout_main.addWidget(self.combo_model_type, 0, 1)

        # 1) Training method
        lbl_training_method = QLabel("Model Type")
        lbl_training_method.setToolTip("The type of model to convert")
        self.layout_main.addWidget(lbl_training_method, 1, 0)

        self.combo_training_method = QComboBox()
        self.combo_training_method.addItem("Base Model", TrainingMethod.FINE_TUNE)
        self.combo_training_method.addItem("LoRA", TrainingMethod.LORA)
        self.combo_training_method.addItem("Embedding", TrainingMethod.EMBEDDING)
        self.layout_main.addWidget(self.combo_training_method, 1, 1)

        # 2) Input name
        lbl_input_name = QLabel("Input name")
        lbl_input_name.setToolTip("Filename, directory or hugging face repository of the base model")
        self.layout_main.addWidget(lbl_input_name, 2, 0)

        self.input_name_line = QLineEdit()
        self.layout_main.addWidget(self.input_name_line, 2, 1)

        # 3) Output data type
        lbl_output_dtype = QLabel("Output Data Type")
        lbl_output_dtype.setToolTip("Precision to use when saving the output model")
        self.layout_main.addWidget(lbl_output_dtype, 3, 0)

        self.combo_output_dtype = QComboBox()
        self.combo_output_dtype.addItem("float32", DataType.FLOAT_32)
        self.combo_output_dtype.addItem("float16", DataType.FLOAT_16)
        self.combo_output_dtype.addItem("bfloat16", DataType.BFLOAT_16)
        self.layout_main.addWidget(self.combo_output_dtype, 3, 1)

        # 4) Output format
        lbl_output_fmt = QLabel("Output Format")
        lbl_output_fmt.setToolTip("Format to use when saving the output model")
        self.layout_main.addWidget(lbl_output_fmt, 4, 0)

        self.combo_output_fmt = QComboBox()
        self.combo_output_fmt.addItem("Safetensors", ModelFormat.SAFETENSORS)
        self.combo_output_fmt.addItem("Diffusers", ModelFormat.DIFFUSERS)
        self.combo_output_fmt.addItem("Checkpoint", ModelFormat.CKPT)
        self.layout_main.addWidget(self.combo_output_fmt, 4, 1)

        # 5) Output model destination
        lbl_output_dest = QLabel("Model Output Destination")
        lbl_output_dest.setToolTip("Filename or directory where the output model is saved")
        self.layout_main.addWidget(lbl_output_dest, 5, 0)

        self.output_dest_line = QLineEdit()
        self.layout_main.addWidget(self.output_dest_line, 5, 1)

        # 6) Convert button
        self.button_convert = QPushButton("Convert")
        self.button_convert.clicked.connect(self.convert_model)
        self.layout_main.addWidget(self.button_convert, 6, 1)

    def convert_model(self):
        """
        Replicates your `convert_model(...)` logic, loading the model, saving,
        handling exceptions, etc.
        """
        try:
            self.button_convert.setEnabled(False)

            # Fill self.convert_model_args from UI
            self.convert_model_args.model_type = self.combo_model_type.currentData()
            self.convert_model_args.training_method = self.combo_training_method.currentData()
            self.convert_model_args.input_name = self.input_name_line.text()
            self.convert_model_args.output_dtype = self.combo_output_dtype.currentData()
            self.convert_model_args.output_model_format = self.combo_output_fmt.currentData()
            self.convert_model_args.output_model_destination = self.output_dest_line.text()

            # Create model loader / saver
            model_loader = create.create_model_loader(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )
            model_saver = create.create_model_saver(
                model_type=self.convert_model_args.model_type,
                training_method=self.convert_model_args.training_method
            )

            print(f"Loading model {self.convert_model_args.input_name}")
            if self.convert_model_args.training_method in [TrainingMethod.FINE_TUNE]:
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        base_model=self.convert_model_args.input_name
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes()
                )
            elif self.convert_model_args.training_method in [TrainingMethod.LORA, TrainingMethod.EMBEDDING]:
                # in your code, you used a uuid4 for the embedding name
                emb_name = str(uuid.uuid4())
                model = model_loader.load(
                    model_type=self.convert_model_args.model_type,
                    model_names=ModelNames(
                        lora=self.convert_model_args.input_name,
                        embedding=EmbeddingName(emb_name, self.convert_model_args.input_name)
                    ),
                    weight_dtypes=self.convert_model_args.weight_dtypes()
                )
            else:
                raise Exception("Could not load model: " + self.convert_model_args.input_name)

            print(f"Saving model {self.convert_model_args.output_model_destination}")
            model_saver.save(
                model=model,
                model_type=self.convert_model_args.model_type,
                output_model_format=self.convert_model_args.output_model_format,
                output_model_destination=self.convert_model_args.output_model_destination,
                dtype=self.convert_model_args.output_dtype.torch_dtype()
            )
            print("Model converted")

        except Exception as e:
            traceback.print_exc()

        torch_gc()
        self.button_convert.setEnabled(True)
