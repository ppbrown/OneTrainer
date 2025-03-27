
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QPushButton, QScrollArea, QFrame, QVBoxLayout
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class OffloadingWindow(QDialog):
    def __init__(
        self,
        parent,
        config: TrainConfig,
        ui_state: UIState,
        *args,
        **kwargs
    ):
        super().__init__(parent, *args, **kwargs)

        self.config = config
        self.ui_state = ui_state

        self.setWindowTitle("Offloading")
        self.resize(800, 400)
        # For a blocking dialog, do self.setModal(True) if you wish

        # Main layout
        self.main_layout = QGridLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.setLayout(self.main_layout)

        # A QScrollArea to replicate ctk.CTkScrollableFrame
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area, 0, 0, 1, 1)

        # We'll create a container QFrame inside the scroll area
        self.container = QFrame()
        self.container_layout = QGridLayout(self.container)
        self.container_layout.setContentsMargins(5, 5, 5, 5)
        self.container_layout.setSpacing(5)
        self.container.setLayout(self.container_layout)
        self.scroll_area.setWidget(self.container)

        # Add the offloading fields
        self.__content_frame(self.container)

        # "OK" button at row=1
        self.ok_button = QPushButton("ok", self)
        self.ok_button.clicked.connect(self.__ok)
        self.main_layout.addWidget(self.ok_button, 1, 0, alignment=Qt.AlignRight)

    def __content_frame(self, parent_frame: QFrame):

        # row=0 => gradient checkpointing
        components.label(parent_frame, 0, 0, "Gradient checkpointing",
                         tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        components.options(
            parent_frame, 0, 1,
            [str(x) for x in list(GradientCheckpointingMethod)],
            self.ui_state,
            "gradient_checkpointing"
        )

        # row=1 => async offloading
        components.label(parent_frame, 1, 0, "Async Offloading", tooltip="Enables Asynchronous offloading.")
        components.switch(parent_frame, 1, 1, self.ui_state, "enable_async_offloading")

        # row=2 => activation offloading
        components.label(parent_frame, 2, 0, "Offload Activations", tooltip="Enables Activation Offloading")
        components.switch(parent_frame, 2, 1, self.ui_state, "enable_activation_offloading")

        # row=3 => layer offload fraction
        components.label(parent_frame, 3, 0, "Layer offload fraction",
                         tooltip=("Enables offloading of individual layers during training to reduce VRAM usage. "
                                  "Increases training time and uses more RAM. "
                                  "Only available if checkpointing is set to CPU_OFFLOADED. "
                                  "Values between 0 and 1, 0=disabled"))
        components.entry(parent_frame, 3, 1, self.ui_state, "layer_offload_fraction")

    def __ok(self):
        self.close()  # or self.accept() to close this dialog
