
# PySide widget that allows a user to browse directories, selecting one file
# at a time. Trigger a generic callback on click, giving it directory name
# and filename seperately

import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QSplitter, QTreeView, QListView, QLineEdit,
    QMainWindow, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, QDir
from PySide6.QtWidgets import QFileSystemModel

class DirectoryBrowser(QWidget):
    def __init__(self, file_clicked_callback, parent=None):
        super().__init__(parent)
        # Ensure the directory browser has a minimum width of 200 pixels.
        self.setMinimumWidth(200)
        self.file_clicked_callback = file_clicked_callback

        # Use a vertical splitter for the directory tree and file list.
        splitter = QSplitter(Qt.Vertical, self)

        # Create two QFileSystemModels:
        # One for the tree view (directories only)...
        self.tree_model = QFileSystemModel(self)
        self.tree_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        # ...and one for the file list (directories and files).
        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(QDir.Files | QDir.NoDotAndDotDot)

        # Start at the system root so users can browse anywhere.
        initial_path = QDir.rootPath()
        self.tree_model.setRootPath(initial_path)
        self.file_model.setRootPath(initial_path)

        # Set up the directory tree view.
        self.tree_view = QTreeView(splitter)
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setRootIndex(self.tree_model.index(initial_path))
        # Hide non-essential columns.
        self.tree_view.setHeaderHidden(True)
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)

        # Set up the file list view.
        self.list_view = QListView(splitter)
        self.list_view.setModel(self.file_model)
        self.list_view.setRootIndex(self.file_model.index(initial_path))
        # Connect file clicks.
        self.list_view.clicked.connect(self.on_file_clicked)

        # QLineEdit for displaying and editing the current directory.
        self.path_display = QLineEdit(initial_path, self)
        self.path_display.returnPressed.connect(self.on_path_entered)

        # Set up the overall layout.
        layout = QVBoxLayout(self)
        layout.addWidget(self.path_display)
        layout.addWidget(splitter)
        self.setLayout(layout)

        # When the user selects a directory from the tree view,
        # update the file list view and the path display.
        self.tree_view.selectionModel().currentChanged.connect(self.on_directory_changed)

    def on_directory_changed(self, current, previous):
        # Retrieve the selected directory path.
        selected_dir = self.tree_model.filePath(current)
        # Update the QLineEdit.
        self.path_display.setText(selected_dir)
        # Update the file list view.
        new_root = self.file_model.setRootPath(selected_dir)
        self.list_view.setRootIndex(new_root)
        # this wont work with its filters
        #self.list_view.setRootIndex(self.file_model.index(selected_dir))

    # User has directly typed/pasted a path in
    def on_path_entered(self):
        new_path = self.path_display.text()
        # On Windows, ensure the path ends with a backslash if it's a drive letter.
        import platform
        if platform.system() == "Windows" and len(new_path) == 2 and new_path[1] == ":":
            new_path += "\\"
        if QDir(new_path).exists():
            # Update both models by calling setRootPath and using the returned index.
            tree_index = self.tree_model.setRootPath(new_path)
            self.tree_view.setRootIndex(tree_index)
            file_index = self.file_model.setRootPath(new_path)
            self.list_view.setRootIndex(file_index)
        else:
            QMessageBox.warning(self, "Invalid Directory",
                                f"The directory '{new_path}' does not exist.")


    def on_file_clicked(self, index):
        # Only proceed if the clicked item is not a directory.
        if not self.file_model.isDir(index):
            full_path = self.file_model.filePath(index)
            directory = os.path.dirname(full_path)
            file_name = os.path.basename(full_path)
            # Call the provided callback with the directory and filename.
            if self.file_clicked_callback:
                self.file_clicked_callback(directory, file_name)


