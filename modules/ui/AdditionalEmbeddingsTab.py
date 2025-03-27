

# This is the "Additional Embeddings" window.
# The tab used in directly training embeddings themselves,
# is currently created in TrainUI.py

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QLineEdit, QCheckBox,
    QGridLayout
)
from PySide6.QtCore import Qt

from modules.util.ui import components

from modules.ui.OTConfigFrame import OTConfigFrame
from modules.util.config.TrainConfig import TrainConfig
from modules.util.config.TrainConfig import TrainEmbeddingConfig
from modules.util.ui.UIState import UIState


class AdditionalEmbeddingsTab(OTConfigFrame):
    def __init__(self, parent, train_config, ui_state):

        super().__init__(
            master=parent,
            train_config=train_config,
            ui_state=ui_state,
            attr_name="additional_embeddings",
            add_button_text="add embedding",
            from_external_file=False,
            is_full_width=True
        )

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

    def refresh_ui(self):
        """
        Overridden method that re-creates the widget list from the config.
        """
        self.__create_element_list()

    def create_widget(self, parent_widget, element, i, open_command, remove_command, clone_command, save_command):
        return EmbeddingWidget(parent_widget, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        """
        Returns a default dictionary for a new embedding.
        """
        return TrainEmbeddingConfig.default_values()

    
    # Typically for when an "Element" needs a detailed dialog for config.
    # We dont need that
    def open_element_window(self, i, ui_state):
        pass


class EmbeddingWidget(QFrame):
    """
    Displays row for each embedding:
      top_frame: close [+], base embedding, placeholder, token_count
      bottom_frame: train, output embedding, stop training after, etc.
    """
    def __init__(self, parent, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__()

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        # QFrame config
        self.setFrameShape(QFrame.StyledPanel)
        # If you want a "rounded" background, you'd do QSS.
        # self.setStyleSheet("background-color: transparent; border-radius: 10px;")

        # We'll use a top-level QGridLayout with two rows:
        #   row=0 => top_frame
        #   row=1 => bottom_frame
        self.layout_grid = QGridLayout(self)
        self.layout_grid.setContentsMargins(5, 5, 5, 5)
        self.layout_grid.setSpacing(5)
        self.setLayout(self.layout_grid)

        # top_frame
        self.top_frame = QFrame(self)
        self.top_frame_layout = QGridLayout(self.top_frame)
        self.top_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.top_frame_layout.setSpacing(5)
        self.layout_grid.addWidget(self.top_frame, 0, 0)

        # bottom_frame
        self.bottom_frame = QFrame(self)
        self.bottom_frame_layout = QGridLayout(self.bottom_frame)
        self.bottom_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_frame_layout.setSpacing(5)
        self.layout_grid.addWidget(self.bottom_frame, 1, 0)

        top_frame = self.top_frame
        bottom_frame = self.bottom_frame
        
        # Top row
        # Close button (X)
        close_button = components.button(top_frame, 0, 0, "X", lambda: remove_command(self.i), "clone")
        close_button.setStyleSheet("background-color: #C00000; color: white; border-radius:2px;")
        close_button.setFixedSize(20, 20)

        # Clone button (+)
        clone_button = components.button(top_frame, 0, 1, "+", lambda: clone_command(self.i, self.__randomize_uuid), "clone")
        clone_button.setStyleSheet("background-color: #00C000; color: white; border-radius:2px;")
        clone_button.setFixedSize(20, 20)
                
        

        # embedding model names
        components.label(top_frame, 0, 2, "base embedding:",
                         tooltip="The base embedding to train on. Leave empty to create a new embedding")
        components.file_entry(
            top_frame, 0, 3, self.ui_state, "model_name",
            path_modifier=lambda x: Path(x).parent.absolute() if x.endswith(".json") else x
        )

        # placeholder
        components.label(top_frame, 0, 4, "placeholder:",
                         tooltip="The placeholder used when using the embedding in a prompt")
        components.entry(top_frame, 0, 5, self.ui_state, "placeholder")

        # token count
        components.label(top_frame, 0, 6, "token count:",
                         tooltip="The token count used when creating a new embedding. Leave empty to auto detect from the initial embedding text.")
        token_count_entry = components.entry(top_frame, 0, 7, self.ui_state, "token_count")

        # trainable
        components.label(bottom_frame, 0, 0, "train:")
        trainable_switch = components.switch(bottom_frame, 0, 1, self.ui_state, "train")

        # output embedding
        components.label(bottom_frame, 0, 2, "output embedding:",
                         tooltip="Output embeddings are calculated at the output of the text encoder, not the input. This can improve results for larger text encoders and lower VRAM usage.")
        output_embedding_switch = components.switch(bottom_frame, 0, 3, self.ui_state, "is_output_embedding")

        # stop training after
        components.label(bottom_frame, 0, 4, "stop training after:",
                         tooltip="When to stop training the embedding")
        components.time_entry(bottom_frame, 0, 5, self.ui_state, "stop_training_after", "stop_training_after_unit")

        # initial embedding text
        components.label(bottom_frame, 0, 6, "initial embedding text:",
                         tooltip="The initial embedding text used when creating a new embedding")
        components.entry(bottom_frame, 0, 7, self.ui_state, "initial_embedding_text")

    def __randomize_uuid(self, embedding_config):
        """
        Cloning logic: randomize the 'uuid' field
        """
        embedding_config.uuid = type(embedding_config).default_values().uuid
        return embedding_config

    def configure_element(self):
        """
        Called if the element changes externally. Refresh the fields.
        """
        self.base_embed_edit.setText(str(self.element.model_name or ""))
        self.placeholder_edit.setText(str(self.element.placeholder or ""))
        if self.element.token_count is not None:
            self.token_count_edit.setText(str(self.element.token_count))
        else:
            self.token_count_edit.clear()

        self.train_switch.setChecked(bool(self.element.train))
        self.output_embedding_switch.setChecked(bool(self.element.is_output_embedding))
        if self.element.stop_training_after is not None:
            self.stop_time_edit.setText(str(self.element.stop_training_after))
        else:
            self.stop_time_edit.clear()

        self.init_text_edit.setText(str(self.element.initial_embedding_text or ""))

    # Obsolete method
    def place_in_list(self):
        """
        Replicates the .grid(...) usage. 
        In Qt, typically the parent layout does `layout.addWidget(self, row, col)`.
        We'll just keep a stub.
        """
        if self.parentWidget() and hasattr(self.parentWidget(), 'layout'):
            parent_layout = self.parentWidget().layout()
            if isinstance(parent_layout, QGridLayout):
                parent_layout.addWidget(self, self.i, 0, 1, 1)
