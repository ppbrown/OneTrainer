# top_bar.py

import json
import os
import traceback
import webbrowser
from collections.abc import Callable
from contextlib import suppress

from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QPushButton, QComboBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.optimizer_util import change_optimizer
from modules.util.ui.UIState import UIState
from modules.util.ui.components import app_title


class TopBar(QWidget):
    def __init__(
        self,
        master,  # in Qt, master is typically the parent, but we'll keep it for signature
        train_config: TrainConfig,
        ui_state: UIState,
        change_model_type_callback: Callable[[ModelType], None],
        change_training_method_callback: Callable[[TrainingMethod], None],
        load_preset_callback: Callable[[], None]
    ):
        super().__init__()  # previously called QWidget.__init__(parent=master) ??

        self.train_config = train_config
        self.ui_state = ui_state
        self.change_model_type_callback = change_model_type_callback
        self.change_training_method_callback = change_training_method_callback
        self.load_preset_callback = load_preset_callback

        # Directory for your presets
        self.dir = "training_presets"

        # This mirrors your "config_ui_data" usage
        self.config_ui_data = {
            "config_name": path_util.canonical_join(self.dir, "#.json")
        }
        self.config_ui_state = UIState(self, self.config_ui_data)

        self.configs = [("", path_util.canonical_join(self.dir, "#.json"))]
        self.__load_available_config_names()

        self.current_config = []

        # We'll build a QGridLayout on 'self'
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        self.setLayout(layout)

        self.training_method_widget = None

        # 1) Title: previously components.app_title(...)
        #self.title_label = QLabel("OneTrainer")  
        self.title_label = app_title()
        # A big font or style might replicate an app title
        layout.addWidget(self.title_label, 0, 0)

        # 2) Configs dropdown (mimics ctk OptionMenu with self.__create_configs_dropdown())
        # We create it after we load the config list
        self.configs_dropdown = None
        self.__create_configs_dropdown()
        layout.addWidget(self.configs_dropdown, 0, 1)

        # 3) "Wiki" button (column 4)
        wiki_button = QPushButton("Wiki")
        wiki_button.clicked.connect(self.open_wiki)
        wiki_button.setToolTip("Open the OneTrainer Wiki in your browser")
        layout.addWidget(wiki_button, 0, 4)

        # 4) "save current config" button (column 3)
        save_button = QPushButton("save current config")
        save_button.setToolTip("Save the current configuration into the training preset directory")
        save_button.clicked.connect(self.__save_config)
        layout.addWidget(save_button, 0, 3)

        # Add some stretch to column 5, to mimic `frame.grid_columnconfigure(5, weight=1)`
        layout.setColumnStretch(5, 1)

        # 5) Model Type: previously components.options_kv(...)
        # We'll create a QComboBox with your list of model types
        self.model_type_combo = QComboBox()
        model_type_values = [
            ("Stable Diffusion 1.5", ModelType.STABLE_DIFFUSION_15),
            ("Stable Diffusion 1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("Stable Diffusion 2.0", ModelType.STABLE_DIFFUSION_20),
            ("Stable Diffusion 2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("Stable Diffusion 2.1", ModelType.STABLE_DIFFUSION_21),
            ("Stable Diffusion 3", ModelType.STABLE_DIFFUSION_3),
            ("Stable Diffusion 3.5", ModelType.STABLE_DIFFUSION_35),
            ("Stable Diffusion XL 1.0 Base", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("Stable Diffusion XL 1.0 Base Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ("Flux Dev", ModelType.FLUX_DEV_1),
            ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
            ("Sana", ModelType.SANA),
            ("Hunyuan Video", ModelType.HUNYUAN_VIDEO)
        ]
        for display_text, enum_val in model_type_values:
            self.model_type_combo.addItem(display_text, enum_val)

        # Set the current index from the config, if we want
        # We'll do a small helper function:
        self.__sync_model_type()
        # Listen for changes
        self.model_type_combo.currentIndexChanged.connect(self.__on_model_type_changed)
        layout.addWidget(self.model_type_combo, 0, 6)

        # Then create the training method combo in col 7
        self.__create_training_method()
        if self.training_method_widget:
            layout.addWidget(self.training_method_widget, 0, 7)

    def __sync_model_type(self):
        """
        Initialize the model type combo from self.ui_state or self.train_config
        """
        current_model_type = self.train_config.model_type
        # find the matching index
        for i in range(self.model_type_combo.count()):
            data = self.model_type_combo.itemData(i)
            if data == current_model_type:
                self.model_type_combo.setCurrentIndex(i)
                return

    def __on_model_type_changed(self, index: int):
        enum_val = self.model_type_combo.itemData(index)
        if enum_val is None:
            return
        self.__change_model_type(enum_val)

    # Might be better named as __create_training_method_combobox
    def __create_training_method(self):
        # If we already had a combobox, remove it
        # Although might be better to just enable/disable the VAE option if possible,
        # since efffectively thats all we really do here.
        if self.training_method_widget:
            self.training_method_widget.deleteLater()
            self.training_method_widget = None

        values = []
        mt = self.train_config.model_type
        if mt.is_stable_diffusion():
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
                ("Fine Tune VAE", TrainingMethod.FINE_TUNE_VAE),
            ]
        elif (
            mt.is_stable_diffusion_3()
            or mt.is_stable_diffusion_xl()
            or mt.is_wuerstchen()
            or mt.is_pixart()
            or mt.is_flux()
            or mt.is_sana()
            or mt.is_hunyuan_video()
        ):
            values = [
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
            ]

        combo = QComboBox()
        for text, method_val in values:
            combo.addItem(text, method_val)

        # set from self.train_config.training_method (if we want)
        current_method = self.train_config.training_method
        # find item that matches current_method
        for i in range(combo.count()):
            if combo.itemData(i) == current_method:
                combo.setCurrentIndex(i)
                break

        # connect
        combo.currentIndexChanged.connect(
            lambda idx: self.change_training_method_callback(combo.itemData(idx))
        )

        self.training_method_widget = combo

    def __change_model_type(self, model_type: ModelType):
        self.change_model_type_callback(model_type)
        # Recreate the training method combo:
        self.__create_training_method()

        # If the widget is in the layout, re-insert it at the same spot
        # We'll assume col=7
        if self.training_method_widget:
            self.layout().addWidget(self.training_method_widget, 0, 7)

    def __create_configs_dropdown(self):
        # We'll replace your "components.options_kv" usage with a QComboBox
        dropdown = QComboBox()
        # Fill from self.configs
        for display_text, path_val in self.configs:
            if not display_text:
                display_text = "#"
            dropdown.addItem(display_text, path_val)

        # Listen for changes
        dropdown.currentIndexChanged.connect(self.__on_config_changed)

        self.configs_dropdown = dropdown

        # set current from config_ui_state
        canonical_path = self.config_ui_state.get_var("config_name").get()
        # find item with that data
        for i in range(dropdown.count()):
            data = dropdown.itemData(i)
            if data == canonical_path:
                dropdown.setCurrentIndex(i)
                break

    def __on_config_changed(self, index: int):
        filename = self.configs_dropdown.itemData(index)
        self.__load_current_config(filename)

    def __load_available_config_names(self):
        if os.path.isdir(self.dir):
            for path in os.listdir(self.dir):
                if path != "#.json":
                    fullpath = path_util.canonical_join(self.dir, path)
                    if fullpath.endswith(".json") and os.path.isfile(fullpath):
                        name = os.path.splitext(os.path.basename(fullpath))[0]
                        self.configs.append((name, fullpath))
            # sort
            self.configs.sort(key=lambda x: x[0])  # sort by display name

    def __save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.train_config.to_settings_dict(secrets=False), f, indent=4)

        return path

    def __save_secrets(self, path) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.train_config.secrets.to_dict(), f, indent=4)
        return path

    def open_wiki(self):
        webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki", new=0, autoraise=False)

    def __save_new_config(self, name):
        path = self.__save_to_file(name)

        # check if it's new
        is_new_config = name not in [x[0] for x in self.configs]
        if is_new_config:
            self.configs.append((name, path))
            self.configs.sort(key=lambda x: x[0])

        # update the UIState config_name if it changed
        canonical_path = path_util.canonical_join(self.dir, f"{name}.json")
        current_stored_path = self.config_ui_state.get_var("config_name").get()
        if current_stored_path != canonical_path:
            self.config_ui_state.get_var("config_name").set(canonical_path)

        if is_new_config:
            # re-create dropdown so it shows the new config
            self.__create_configs_dropdown()
            # re-insert into the layout if needed
            self.layout().addWidget(self.configs_dropdown, 0, 1)

    def __save_config(self):
        default_value = self.configs_dropdown.currentText()
        # remove leading "#"
        while default_value.startswith('#'):
            default_value = default_value[1:]

        # Instead of dialogs.StringInputDialog, 
        # we can do a quick pop-up QInputDialog if you want. 
        # For demonstration, let's do a minimal approach:
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Config Name", "Enter new config name:", text=default_value)
        if ok and new_name and not new_name.startswith("#"):
            self.__save_new_config(new_name)

    def __load_current_config(self, filename):
        if not filename:
            return
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r", encoding="utf-8") as f:
                loaded_dict = json.load(f)
                default_config = TrainConfig.default_values()
                if is_built_in_preset:
                    loaded_dict["__version"] = default_config.config_version
                loaded_config = default_config.from_dict(loaded_dict).to_unpacked_config()

            with suppress(FileNotFoundError):
                with open("secrets.json", "r", encoding="utf-8") as f:
                    secrets_dict = json.load(f)
                    loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.train_config.from_dict(loaded_config.to_dict())
            self.ui_state.update(loaded_config)

            optimizer_config = change_optimizer(self.train_config)
            self.ui_state.get_var("optimizer").update(optimizer_config)

            self.load_preset_callback()
        except FileNotFoundError:
            pass
        except Exception:
            print(traceback.format_exc())

    def __remove_config(self):
        # TODO
        pass

    def save_default(self):
        self.__save_to_file("#")
        self.__save_secrets("secrets.json")
