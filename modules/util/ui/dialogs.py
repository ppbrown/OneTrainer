# string_input_dialog.py

"""
XXX Check places that we use this...

chatgpt says we could potentially replace this whole class:

QInputDialog – specifically its static method QInputDialog.getText(...) – can serve as a straightforward replacement. It displays a small dialog with a single text field, an “OK” and “Cancel” button, an optional default value, and returns the user’s input.\
     For example:


from PySide6.QtWidgets import QInputDialog, QLineEdit

text, ok = QInputDialog.getText(
    parent=self,
    title="Your Title",
    label="Enter something:",
    echo=QLineEdit.Normal,
    text="default value"
)

if ok:
    print("User typed:", text)

    
"""

from typing import Callable, Optional

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QLabel, QLineEdit, QPushButton
)
from PySide6.QtCore import Qt


class StringInputDialog(QDialog):

    def __init__(
        self,
        parent,
        title: str,
        question: str,
        callback: Callable[[str], None],
        default_value: Optional[str] = None,
        validate_callback: Optional[Callable[[str], bool]] = None,
        *args,
        **kwargs
    ):
        super().__init__(parent, *args, **kwargs)

        self.callback = callback
        self.validate_callback = validate_callback

        self.setWindowTitle(title)
        self.resize(300, 120)
        self.setModal(True)

        # Layout
        layout = QGridLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        self.setLayout(layout)

        # Question label
        self.question_label = QLabel(question, self)
        layout.addWidget(self.question_label, 0, 0, 1, 2)

        # Entry
        self.entry = QLineEdit(self)
        layout.addWidget(self.entry, 1, 0, 1, 2)

        # Ok button
        self.ok_button = QPushButton("ok", self)
        self.ok_button.clicked.connect(self.ok)
        layout.addWidget(self.ok_button, 2, 0, 1, 1)

        # Cancel button
        self.cancel_button = QPushButton("cancel", self)
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button, 2, 1, 1, 1)

        if default_value is not None:
            self.entry.setText(default_value)

    def ok(self):
        text_val = self.entry.text()
        if self.validate_callback is None or self.validate_callback(text_val):
            self.callback(text_val)
            self.close()

    def cancel(self):
        self.close()
