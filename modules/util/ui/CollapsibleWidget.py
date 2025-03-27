

from PySide6.QtWidgets import QApplication, QWidget, QToolButton, QVBoxLayout, QFrame, QTextEdit
from PySide6.QtCore import Qt
import sys

# Provide a widget that can minimize its contents in place,
# if you click on its minimize button.
# Similar in some ways to a ScrollArea, in that its use is what it contains
# We default to QFrame.StyledPanel, but you can always use
# setFraameShape to set a different style if you wish
class CollapsibleWidget(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.contentArea = None
        
        self.setFrameShape(QFrame.StyledPanel)

        # Create a toggle button that will serve as the header.
        self.toggleButton = QToolButton(text=title, checkable=True, checked=True)
        self.toggleButton.setStyleSheet("QToolButton { border: none; }")
        # Display text next to an arrow icon.
        self.toggleButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggleButton.setArrowType(Qt.DownArrow)
        self.toggleButton.clicked.connect(self.on_toggle)

        # Set up the layout for this collapsible widget.
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(self.toggleButton)

    def setWidget(self, widget: QWidget):
        self.contentArea = widget
        self.mainLayout.addWidget(widget)

    def widget(self) -> QWidget:
        return contentArea


    def on_toggle(self, checked):
        if not self.contentArea:
            return
        self.contentArea.setVisible(checked)
        self.toggleButton.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
    


if __name__ == "__main__":

    # Example usage with two collapsible sub-widgets.
    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)

    contentArea = QFrame()
    contentArea.setFrameShape(QFrame.NoFrame)
    contentLayout = QVBoxLayout(contentArea)
    contentLayout.setContentsMargins(15, 0, 0, 0)

    # Sample widget 1
    dataWidget = CollapsibleWidget("data")
    dataWidget.setWidget(QTextEdit("Data subwidget content"))
    
    # sample widget 2
    processingWidget = CollapsibleWidget("processing")
    processingWidget.setWidget(QTextEdit("Processing subwidget content"))

    layout.addWidget(dataWidget)
    layout.addWidget(processingWidget)

    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())
