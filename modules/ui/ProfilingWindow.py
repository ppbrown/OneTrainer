
import faulthandler

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt, QEvent

from scalene import scalene_profiler


class ProfilingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Profiling")
        self.resize(512, 512)

        # If you want the window to hide instead of truly close when the user clicks X:
        # In Tk, we did "self.protocol('WM_DELETE_WINDOW', self.withdraw)"
        # We'll replicate that by overriding closeEvent below.

        # Top-level layout
        main_layout = QGridLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # 2 Buttons in row 0, 1
        self.dump_stack_button = QPushButton("Dump stack")
        self.dump_stack_button.clicked.connect(self._dump_stack)
        main_layout.addWidget(self.dump_stack_button, 0, 0)

        self._profile_button = QPushButton("Start Profiling")
        self._profile_button.setToolTip("Turns on/off Scalene profiling. "
                                        "Only works when OneTrainer is launched with Scalene!")
        self._profile_button.clicked.connect(self._start_profiler)
        main_layout.addWidget(self._profile_button, 1, 0)

        # A bottom bar in row 2
        bottom_bar = QFrame()
        bottom_bar_layout = QHBoxLayout(bottom_bar)
        bottom_bar.setLayout(bottom_bar_layout)

        self._message_label = QLabel("Inactive")
        bottom_bar_layout.addWidget(self._message_label)

        main_layout.addWidget(bottom_bar, 2, 0)

        # Hide by default
        self.hide()

    def closeEvent(self, event):
        """
        Override the close event to hide the window instead of actually closing.
        """
        self.hide()
        event.ignore()

    def _dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(file=f)
        self._message_label.setText('Stack dumped to stacks.txt')

    def _end_profiler(self):
        scalene_profiler.stop()
        self._message_label.setText('Inactive')
        self._profile_button.setText('Start Profiling')
        self._profile_button.clicked.disconnect()
        self._profile_button.clicked.connect(self._start_profiler)

    def _start_profiler(self):
        scalene_profiler.start()
        self._message_label.setText('Profiling active...')
        self._profile_button.setText('End Profiling')
        self._profile_button.clicked.disconnect()
        self._profile_button.clicked.connect(self._end_profiler)
        