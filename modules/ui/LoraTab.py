
# Handle the Lora tab, when present

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QFrame, QGridLayout, QLabel, QComboBox, QLineEdit, QCheckBox
)
from PySide6.QtCore import Qt

from modules.modelSetup.FluxLoRASetup import PRESETS as flux_presets
from modules.modelSetup.HunyuanVideoLoRASetup import PRESETS as hunyuan_video_presets
from modules.modelSetup.PixArtAlphaLoRASetup import PRESETS as pixart_presets
from modules.modelSetup.SanaLoRASetup import PRESETS as sana_presets
from modules.modelSetup.StableDiffusion3LoRASetup import PRESETS as sd3_presets
from modules.modelSetup.StableDiffusionLoRASetup import PRESETS as sd_presets
from modules.modelSetup.StableDiffusionXLLoRASetup import PRESETS as sdxl_presets
from modules.modelSetup.WuerstchenLoRASetup import PRESETS as sc_presets
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import PeftType
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class LoraTab:

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__()
        self.master = master
        self.train_config = train_config
        self.ui_state = ui_state

        self.layer_entry = None
        self.layer_selector = None
        self.presets = {}
        self.presets_list = []
        self.prior_custom = ""
        self.prior_selected = None

        # We'll store references to the main container frames
        self.scroll_frame = None
        self.options_frame = None

        self.refresh_ui()

    def refresh_ui(self):
        # If we already built a frame, remove it
        if self.scroll_frame:
            self.scroll_frame.setParent(None)
            self.scroll_frame = None

        # A QFrame as root container
        self.scroll_frame = QFrame(self.master)
        self.scroll_frame_layout = QGridLayout(self.scroll_frame)
        self.scroll_frame_layout.setContentsMargins(0,0,0,0)
        self.scroll_frame_layout.setSpacing(5)
        self.scroll_frame.setLayout(self.scroll_frame_layout)

        # If 'master' has a layout, we can add this widget
        if hasattr(self.master, 'layout') and self.master.layout():
            self.master.layout().addWidget(self.scroll_frame)
        else:
            # Or place it in a grid if your 'master' is also a container
            self.scroll_frame.setGeometry(0, 0, 300, 200)  # fallback geometry

        # Determine which LoRA presets we should use
        if self.train_config.model_type.is_stable_diffusion():
            self.presets = sd_presets
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.presets = sdxl_presets
        elif self.train_config.model_type.is_stable_diffusion_3():
            self.presets = sd3_presets
        elif self.train_config.model_type.is_wuerstchen():
            self.presets = sc_presets
        elif self.train_config.model_type.is_pixart():
            self.presets = pixart_presets
        elif self.train_config.model_type.is_flux():
            self.presets = flux_presets
        elif self.train_config.model_type.is_sana():
            self.presets = sana_presets
        elif self.train_config.model_type.is_hunyuan_video():
            self.presets = hunyuan_video_presets
        else:
            self.presets = {"full": []}

        self.presets_list = list(self.presets.keys()) + ["custom"]

        # "Type" label + options_kv => a QComboBox that calls self.setup_lora
        components.label(self.scroll_frame, 0, 0, "Type",
                         tooltip="The type of low-parameter finetuning method.")
        components.options_kv(
            self.scroll_frame, 0, 1,
            [
                ("LoRA", PeftType.LORA),
                ("LoHa", PeftType.LOHA),
            ],
            self.ui_state,
            "peft_type",
            command=self.setup_lora
        )

    def setup_lora(self, peft_type: PeftType):
        name = "LoHa" if peft_type == PeftType.LOHA else "LoRA"

        # Remove the old frame if present
        if self.options_frame:
            self.options_frame.setParent(None)
            self.options_frame = None

        self.options_frame = QFrame(self.scroll_frame)
        options_layout = QGridLayout(self.options_frame)
        options_layout.setContentsMargins(0,0,0,0)
        options_layout.setSpacing(5)
        self.options_frame.setLayout(options_layout)

        # place at row=1 col=0..3
        self.scroll_frame_layout.addWidget(self.options_frame, 1, 0, 1, 3)

        # row=0 => lora model name
        components.label(self.options_frame, 0, 0, f"{name} base model",
                         tooltip=f"The base {name} to train on. "
                                 "Leave empty to create a new LoRA")
        entry = components.file_entry(
            self.options_frame, 0, 1,
            self.ui_state, "lora_model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )
        # expand entry over columns 1..4
        entry.grid(row=0, column=1, columnspan=4)

        # row=1 => lora rank
        components.label(self.options_frame, 1, 0, f"{name} rank",
                         tooltip=f"The rank parameter used when creating a new {name}")
        components.entry(self.options_frame, 1, 1, self.ui_state, "lora_rank")

        if peft_type == PeftType.LORA:
            # row=1 => col=3 => "Decompose Weights" switch
            components.label(self.options_frame, 1, 3, "Decompose Weights (DoRA)",
                             tooltip="Decompose LoRA Weights (aka, DoRA).")
            components.switch(self.options_frame, 1, 4, self.ui_state, "lora_decompose")

            # row=2 => col=3 => "Use Norm Epsilon"
            components.label(self.options_frame, 2, 3, "Use Norm Espilon (DoRA Only)",
                             tooltip="Add an epsilon to the norm division calculation in DoRA. "
                                     "Can help with stability.")
            components.switch(self.options_frame, 2, 4, self.ui_state, "lora_decompose_norm_epsilon")

        # row=2 => lora alpha
        components.label(self.options_frame, 2, 0, f"{name} alpha",
                         tooltip=f"The alpha parameter used when creating a new {name}")
        components.entry(self.options_frame, 2, 1, self.ui_state, "lora_alpha")

        # row=3 => dropout
        components.label(self.options_frame, 3, 0, "Dropout Probability",
                         tooltip="Dropout probability. 0 disables, 1 = maximum.")
        components.entry(self.options_frame, 3, 1, self.ui_state, "dropout_probability")

        # row=4 => lora weight dtype
        components.label(self.options_frame, 4, 0, f"{name} Weight Data Type",
                         tooltip=f"The {name} weight dtype for training. Reduces memory consumption but also precision.")
        components.options_kv(
            self.options_frame, 4, 1,
            [
                ("float32", DataType.FLOAT_32),
                ("bfloat16", DataType.BFLOAT_16),
            ],
            self.ui_state,
            "lora_weight_dtype"
        )

        # row=5 => bundle embeddings
        components.label(self.options_frame, 5, 0, "Bundle Embeddings",
                         tooltip=f"Bundles additional embeddings into the {name} output file.")
        components.switch(self.options_frame, 5, 1, self.ui_state, "bundle_additional_embeddings")

        # row=6 => layer preset
        components.label(self.options_frame, 6, 0, "Layer Preset",
                         tooltip="Select a preset or 'Custom' to define your own layers.")
        self.layer_selector = components.options(
            self.options_frame, 6, 1, self.presets_list, self.ui_state, "lora_layer_preset",
            command=self.__preset_set_layer_choice
        )

        self.layer_entry = components.entry(
            self.options_frame, 6, 2, self.ui_state, "lora_layers",
            tooltip=f"Comma-separated list of diffusion layers for {name}."
        )
        # store user-defined layers
        self.prior_custom = self.train_config.lora_layers or ""
        self.layer_entry.grid(row=6, column=2, columnspan=3, sticky="ew")

        # If the user’s config had a preset that's not in the list, default to the first
        if self.layer_selector.get() not in self.presets_list:
            self.layer_selector.set(self.presets_list[0])
        self.__preset_set_layer_choice(self.layer_selector.get())

    def __preset_set_layer_choice(self, selected: str):
        if not selected:
            selected = self.presets_list[0]

        if selected == "custom":
            self.layer_entry.configure(state="normal")
            # set the user-specified custom layers
            self.layer_entry.cget('textvariable').set(self.prior_custom)
        else:
            # If switching away from "custom," remember what the user typed
            if self.prior_selected == "custom":
                self.prior_custom = self.layer_entry.get()
            # Now freeze the entry to read-only
            self.layer_entry.configure(state="readonly")
            # apply the preset
            chosen_layers = self.presets.get(selected, [])
            self.layer_entry.cget('textvariable').set(",".join(chosen_layers))
        self.prior_selected = selected
