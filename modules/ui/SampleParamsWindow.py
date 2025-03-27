
# Allows a user to customize configuration info for a SampleFrame item

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt

from modules.ui.SampleFrame import SampleFrame  # assuming this is the PySide6 version
from modules.util.config.SampleConfig import SampleConfig
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class SampleParamsWindow(QDialog):
    def __init__(self, parent, sample: SampleConfig, ui_state: UIState, *args, **kwargs):
        super().__init__()

        self.sample = sample
        self.ui_state = ui_state

        self.setWindowTitle("Sample")
        self.resize(800, 500)
        self.setModal(False)

        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # SampleFrame in row=0
        sample_frame = SampleFrame(self, self.sample, self.ui_state)
        main_layout.addWidget(sample_frame, 0, 0, 1, 1)

        # "ok" button in row=1
        self.ok_button = components.button(self, 1, 0, "ok", self.__ok)
        main_layout.addWidget(self.ok_button, 1, 0, alignment=Qt.AlignRight)

    def __ok(self):
        self.close()
