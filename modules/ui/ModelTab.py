
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QPushButton
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui.UIState import UIState


class ModelTab(QWidget):

    def __init__(self, parent: QWidget, train_config: TrainConfig, ui_state: UIState):
        super().__init__(parent)

        self.train_config = train_config
        self.ui_state = ui_state

        # We'll hold references for the scroll area
        self.scroll_area = None
        self.scroll_area_widget = None
        self.scroll_area_layout = None

        # top-level layout for this ModelTab
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create the scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # The actual container inside the scroll area
        self.scroll_area_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget)

        self.scroll_area_layout = QGridLayout(self.scroll_area_widget)
        self.scroll_area_layout.setColumnStretch(1, 10)
        self.scroll_area_layout.setColumnStretch(4, 1)
        # The old code used .grid_rowconfigure(0, weight=1), etc. in Tk
        # We approximate with layout.setRowStretch(row, weight)

        # Now do the initial UI creation
        self.refresh_ui()

    def refresh_ui(self):
        """
        Rebuild the UI in the scroll area based on the current model_type.
        We'll remove existing layout items if needed.
        """
        # Clear out old widgets in scroll_area_layout
        # Easiest approach: remove them in reverse order
        while self.scroll_area_layout.count() > 0:
            item = self.scroll_area_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        # Now add fresh widgets
        if self.train_config.model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui()
        elif self.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui()
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui()
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui()
        elif self.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui()
        elif self.train_config.model_type.is_flux():
            self.__setup_flux_ui()
        elif self.train_config.model_type.is_sana():
            self.__setup_sana_ui()
        elif self.train_config.model_type.is_hunyuan_video():
            self.__setup_hunyuan_video_ui()

    # -----------------------------------------------------------------------
    # Each specialized UI
    # -----------------------------------------------------------------------
    def __setup_stable_diffusion_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_unet=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method in [
                TrainingMethod.FINE_TUNE,
                TrainingMethod.FINE_TUNE_VAE,
            ],
            allow_checkpoint=True,
        )

    def __setup_stable_diffusion_3_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_text_encoder_3=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=True,
        )

    def __setup_flux_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=False,
        )

    def __setup_stable_diffusion_xl_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_unet=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=True,
        )

    def __setup_wuerstchen_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            allow_override_prior=self.train_config.model_type.is_stable_cascade(),
            has_text_encoder=True,
        )
        row = self.__create_effnet_encoder_components(row)
        row = self.__create_decoder_components(row, self.train_config.model_type.is_wuerstchen_v2())
        row = self.__create_output_components(
            row,
            allow_safetensors=self.train_config.training_method != TrainingMethod.FINE_TUNE
                              or self.train_config.model_type.is_stable_cascade(),
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=self.train_config.training_method != TrainingMethod.FINE_TUNE,
        )

    def __setup_pixart_alpha_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=True,
        )

    def __setup_sana_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=self.train_config.training_method != TrainingMethod.FINE_TUNE,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=False,
        )

    def __setup_hunyuan_video_ui(self):
        row = 0
        row = self.__create_base_dtype_components(row)
        row = self.__create_base_components(
            row,
            has_prior=True,
            has_text_encoder_1=True,
            has_text_encoder_2=True,
            has_vae=True,
        )
        row = self.__create_output_components(
            row,
            allow_safetensors=True,
            allow_diffusers=self.train_config.training_method == TrainingMethod.FINE_TUNE,
            allow_checkpoint=True,
        )

    # -----------------------------------------------------------------------
    # Common sub-UI pieces
    # -----------------------------------------------------------------------
    def __create_dtype_options(self, include_none: bool = True) -> list[tuple[str, DataType]]:
        """
        Returns a list of (display_text, DataType) for populating a combo box.
        """
        options = [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float16", DataType.FLOAT_16),
            ("float8", DataType.FLOAT_8),
            ("nfloat4", DataType.NFLOAT_4),
            # ("int8", DataType.INT_8), # commented out in your code
        ]
        if include_none:
            options.insert(0, ("", DataType.NONE))
        return options

    def __create_base_dtype_components(self, row: int) -> int:
        # Hugging Face Token
        lbl = QLabel("Hugging Face Token")
        lbl.setToolTip("Enter your Hugging Face access token if you have used a protected Hugging Face repository below.\n"
                       "This value is stored separately, not saved to your configuration file.\n"
                       "Go to https://huggingface.co/settings/tokens to create an access token.")
        self.scroll_area_layout.addWidget(lbl, row, 0)

        hf_line = QLineEdit()
        # If you need to bind this to self.ui_state["secrets.huggingface_token"], do so
        self.scroll_area_layout.addWidget(hf_line, row, 1)

        row += 1

        # base model
        lbl2 = QLabel("Base Model")
        lbl2.setToolTip("Filename, directory or Hugging Face repository of the base model")
        self.scroll_area_layout.addWidget(lbl2, row, 0)

        base_model_line = QLineEdit()
        # If you want a "file dialog" or something, you'd add a button or a custom approach
        self.scroll_area_layout.addWidget(base_model_line, row, 1)

        # weight dtype
        lbl3 = QLabel("Weight Data Type")
        lbl3.setToolTip("The base model weight data type used for training.\nThis can reduce memory usage but reduces precision.")
        self.scroll_area_layout.addWidget(lbl3, row, 3)

        dtype_combo = QComboBox()
        dtype_opts = self.__create_dtype_options(False)
        for text, enum_val in dtype_opts:
            dtype_combo.addItem(text, enum_val)
        self.scroll_area_layout.addWidget(dtype_combo, row, 4)

        row += 1
        return row

    def __create_base_components(
        self,
        row: int,
        has_unet: bool = False,
        has_prior: bool = False,
        allow_override_prior: bool = False,
        has_text_encoder: bool = False,
        has_text_encoder_1: bool = False,
        has_text_encoder_2: bool = False,
        has_text_encoder_3: bool = False,
        has_vae: bool = False,
    ) -> int:

        if has_unet:
            lbl = QLabel("Override UNet Data Type")
            lbl.setToolTip("Overrides the unet weight data type")
            self.scroll_area_layout.addWidget(lbl, row, 3)
            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)
            row += 1

        if has_prior:
            if allow_override_prior:
                # prior model
                lblp = QLabel("Prior Model")
                lblp.setToolTip("Filename, directory or Hugging Face repository of the prior model")
                self.scroll_area_layout.addWidget(lblp, row, 0)

                prior_line = QLineEdit()
                self.scroll_area_layout.addWidget(prior_line, row, 1)
                # or a browse button if you want

            # prior dtype
            lblp2 = QLabel("Override Prior Data Type")
            lblp2.setToolTip("Overrides the prior weight data type")
            self.scroll_area_layout.addWidget(lblp2, row, 3)
            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)
            row += 1

        if has_text_encoder:
            lblt = QLabel("Override Text Encoder Data Type")
            lblt.setToolTip("Overrides the text encoder weight data type")
            self.scroll_area_layout.addWidget(lblt, row, 3)
            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)
            row += 1

        if has_text_encoder_1:
            lblt1 = QLabel("Override Text Encoder 1 Data Type")
            lblt1.setToolTip("Overrides the text encoder 1 weight data type")
            self.scroll_area_layout.addWidget(lblt1, row, 3)
            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)
            row += 1

        if has_text_encoder_2:
            lblt2 = QLabel("Override Text Encoder 2 Data Type")
            lblt2.setToolTip("Overrides the text encoder 2 weight data type")
            self.scroll_area_layout.addWidget(lblt2, row, 3)
            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)
            row += 1

        if has_text_encoder_3:
            lblt3 = QLabel("Override Text Encoder 3 Data Type")
            lblt3.setToolTip("Overrides the text encoder 3 weight data type")
            self.scroll_area_layout.addWidget(lblt3, row, 3)
            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)
            row += 1

        if has_vae:
            lblv = QLabel("VAE Override")
            lblv.setToolTip("Directory or Hugging Face repository of a VAE model in diffusers format.\n"
                            "Can override the VAE included in the base model.\n"
                            "Using a safetensor VAE file might cause issues if not in diffusers format.")
            self.scroll_area_layout.addWidget(lblv, row, 0)

            vae_line = QLineEdit()
            self.scroll_area_layout.addWidget(vae_line, row, 1)

            lblv2 = QLabel("Override VAE Data Type")
            lblv2.setToolTip("Overrides the vae weight data type")
            self.scroll_area_layout.addWidget(lblv2, row, 3)

            combo = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo, row, 4)

            row += 1

        return row

    def __create_effnet_encoder_components(self, row: int) -> int:
        lbl = QLabel("Effnet Encoder Model")
        lbl.setToolTip("Filename, directory or Hugging Face repository of the effnet encoder model")
        self.scroll_area_layout.addWidget(lbl, row, 0)

        line = QLineEdit()
        self.scroll_area_layout.addWidget(line, row, 1)

        lbl2 = QLabel("Override Effnet Encoder Data Type")
        lbl2.setToolTip("Overrides the effnet encoder weight data type")
        self.scroll_area_layout.addWidget(lbl2, row, 3)

        combo = self.__make_dtype_combo(True)
        self.scroll_area_layout.addWidget(combo, row, 4)

        row += 1
        return row

    def __create_decoder_components(self, row: int, has_text_encoder: bool) -> int:
        lbl = QLabel("Decoder Model")
        lbl.setToolTip("Filename, directory or Hugging Face repository of the decoder model")
        self.scroll_area_layout.addWidget(lbl, row, 0)

        dec_line = QLineEdit()
        self.scroll_area_layout.addWidget(dec_line, row, 1)

        lbl2 = QLabel("Override Decoder Data Type")
        lbl2.setToolTip("Overrides the decoder weight data type")
        self.scroll_area_layout.addWidget(lbl2, row, 3)

        combo = self.__make_dtype_combo(True)
        self.scroll_area_layout.addWidget(combo, row, 4)
        row += 1

        if has_text_encoder:
            lbl3 = QLabel("Override Decoder Text Encoder Data Type")
            lbl3.setToolTip("Overrides the decoder text encoder weight data type")
            self.scroll_area_layout.addWidget(lbl3, row, 3)

            combo2 = self.__make_dtype_combo(True)
            self.scroll_area_layout.addWidget(combo2, row, 4)
            row += 1

        lbl4 = QLabel("Override Decoder VQGAN Data Type")
        lbl4.setToolTip("Overrides the decoder vqgan weight data type")
        self.scroll_area_layout.addWidget(lbl4, row, 3)

        combo3 = self.__make_dtype_combo(True)
        self.scroll_area_layout.addWidget(combo3, row, 4)
        row += 1

        return row

    def __create_output_components(
        self,
        row: int,
        allow_safetensors: bool = False,
        allow_diffusers: bool = False,
        allow_checkpoint: bool = False,
    ) -> int:
        # output model destination
        lbl = QLabel("Model Output Destination")
        lbl.setToolTip("Filename or directory where the output model is saved")
        self.scroll_area_layout.addWidget(lbl, row, 0)

        output_line = QLineEdit()
        self.scroll_area_layout.addWidget(output_line, row, 1)

        lbl2 = QLabel("Output Data Type")
        lbl2.setToolTip("Precision to use when saving the output model")
        self.scroll_area_layout.addWidget(lbl2, row, 3)

        combo = QComboBox()
        dtype_opts = [
            ("float16", DataType.FLOAT_16),
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
            ("float8", DataType.FLOAT_8),
            ("nfloat4", DataType.NFLOAT_4),
        ]
        for text, enum_val in dtype_opts:
            combo.addItem(text, enum_val)
        self.scroll_area_layout.addWidget(combo, row, 4)

        row += 1

        # output format
        formats = []
        if allow_safetensors:
            formats.append(("Safetensors", ModelFormat.SAFETENSORS))
        if allow_diffusers:
            formats.append(("Diffusers", ModelFormat.DIFFUSERS))
        if allow_checkpoint:
            formats.append(("Checkpoint", ModelFormat.CKPT))

        lbl3 = QLabel("Output Format")
        lbl3.setToolTip("Format to use when saving the output model")
        self.scroll_area_layout.addWidget(lbl3, row, 0)

        fmt_combo = QComboBox()
        for text, enum_val in formats:
            fmt_combo.addItem(text, enum_val)
        self.scroll_area_layout.addWidget(fmt_combo, row, 1)

        lbl4 = QLabel("Include Config")
        lbl4.setToolTip("Include the training configuration in the final model.\n"
                        "Only supported for safetensors.\n"
                        "None: no config included.\n"
                        "Settings: all training settings.\n"
                        "All: everything (samples, concepts).")
        self.scroll_area_layout.addWidget(lbl4, row, 3)

        cfg_combo = QComboBox()
        cfg_values = [
            ("None", ConfigPart.NONE),
            ("Settings", ConfigPart.SETTINGS),
            ("All", ConfigPart.ALL),
        ]
        for text, enum_val in cfg_values:
            cfg_combo.addItem(text, enum_val)
        self.scroll_area_layout.addWidget(cfg_combo, row, 4)

        row += 1
        return row

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def __make_dtype_combo(self, include_none: bool = True) -> QComboBox:
        """
        Helper to create a QComboBox with dtype options.
        """
        combo = QComboBox()
        options = self.__create_dtype_options(include_none)
        for text, enum_val in options:
            combo.addItem(text, enum_val)
        return combo
