import os
import platform
import subprocess
import traceback

"""
This is a window that gets used under the "Dataset Tools" tab, but 
can also be used as a standalone program, via scripts/caption_ui.py
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QCheckBox, QLineEdit, QPlainTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QKeyEvent

from modules.ui.DirectoryBrowser import DirectoryBrowser

from modules.module.ClipSegModel import ClipSegModel
from modules.module.MaskByColor import MaskByColor
from modules.module.RembgHumanModel import RembgHumanModel
from modules.module.RembgModel import RembgModel
from modules.ui.GenerateCaptionsWindow import GenerateCaptionsWindow
from modules.ui.GenerateMasksWindow import GenerateMasksWindow
from modules.util import path_util
from modules.util.torch_util import default_device

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw

class CaptionUI(QMainWindow):

    def __init__(
        self,
        parent=None,
        initial_dir: str | None = None,
        initial_include_subdirectories: bool = False
    ):
        super().__init__(parent)

        self.setWindowTitle("OneTrainer")
        self.show()
        self.raise_()

        self.dir = initial_dir if initial_dir is not None else ""
        self.config_ui_data = {
            "include_subdirectories": initial_include_subdirectories
        }

        # Note: this size MUST be smaller than screen size or very bad things happen.
        self.image_size = 650
        self.help_text = (
            "Keyboard shortcuts when focusing on the prompt input field:\n"
            "Up arrow: previous image\n"
            "Down arrow: next image\n"
            "Return: save\n"
            "Ctrl+M: only show the mask\n"
            "Ctrl+D: draw mask editing mode\n"
            "Ctrl+F: fill mask editing mode\n\n"
            "When editing masks:\n"
            "Left click: add mask\n"
            "Right click: remove mask\n"
            "Mouse wheel: increase or decrease brush size"
        )

        self.masking_model = None

        self.image_rel_paths = []
        self.current_image_index = -1

        # Image & mask data
        self.pil_image = None
        self.image_width = 0
        self.image_height = 0
        self.pil_mask = None
        self.mask_draw_x = 0
        self.mask_draw_y = 0
        self.mask_draw_radius = 0.01
        self.display_only_mask = False
        self.mask_editing_mode = 'draw'
        self.enable_mask_editing = False

        self.prompt_text = ""
        self.brush_alpha = 1.0

        # -------------------------------------------------------------------
        # Main Layout
        # -------------------------------------------------------------------
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        # Top bar
        self.top_bar_layout = QHBoxLayout()
        main_layout.addLayout(self.top_bar_layout)
        self.setup_top_bar()

        # Bottom area: left column = DirectoryBrowser, right column = content
        self.bottom_splitter = QSplitter(Qt.Horizontal)

        main_layout.addWidget(self.bottom_splitter, stretch=1)

        # Replace the custom QListWidget with a DirectoryBrowser widget.
        # Pass a callback that receives (directory, file_name) when a file is clicked.
        self.directory_browser = DirectoryBrowser(file_clicked_callback=self.on_file_selected)
        self.bottom_splitter.addWidget(self.directory_browser)

        # Center content (image, mask, caption)
        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.bottom_splitter.addWidget(self.content_widget)
        self.setup_content_column()

        # Initialize directory and image list.
        self.load_directory(self.config_ui_data["include_subdirectories"])

    def setup_top_bar(self):
        open_button = QPushButton("Open")
        open_button.setToolTip("Open a new directory")
        open_button.clicked.connect(self.open_directory)
        self.top_bar_layout.addWidget(open_button)

        gen_masks_button = QPushButton("Generate Masks")
        gen_masks_button.setToolTip("Automatically generate masks")
        gen_masks_button.clicked.connect(self.open_mask_window)
        self.top_bar_layout.addWidget(gen_masks_button)

        gen_captions_button = QPushButton("Generate Captions")
        gen_captions_button.setToolTip("Automatically generate captions")
        gen_captions_button.clicked.connect(self.open_caption_window)
        self.top_bar_layout.addWidget(gen_captions_button)

        if platform.system() == "Windows":
            explorer_button = QPushButton("Open in Explorer")
            explorer_button.setToolTip("Open the current image in Explorer")
            explorer_button.clicked.connect(self.open_in_explorer)
            self.top_bar_layout.addWidget(explorer_button)

        self.include_subdirectories_checkbox = QCheckBox("include subdirectories")
        self.include_subdirectories_checkbox.setChecked(self.config_ui_data["include_subdirectories"])
        self.include_subdirectories_checkbox.stateChanged.connect(self.toggle_include_subdirectories)
        self.top_bar_layout.addWidget(self.include_subdirectories_checkbox)

        # Add some stretch to push the "Help" button to the right
        self.top_bar_layout.addStretch(1)

        help_button = QPushButton("Help")
        help_button.setToolTip(self.help_text)
        help_button.clicked.connect(self.print_help)
        self.top_bar_layout.addWidget(help_button)

    # in Qt, state is NOT a boolean, but an integer: 0 (unchecked), 1 (partially checked), 2 (checked)
    def toggle_include_subdirectories(self, state):
        self.config_ui_data["include_subdirectories"] = True if state == 2 else False

    # -----------------------------------------------------------------------
    # File selection callback (from DirectoryBrowser)
    # -----------------------------------------------------------------------
    def on_file_selected(self, directory, file_name):
        # When a file is clicked in the DirectoryBrowser,
        # update the current directory and scan it for supported images.
        self.dir = directory
        self.scan_directory(self.config_ui_data["include_subdirectories"])
        # Find the index in the scanned image list that matches the clicked file.
        selected_index = -1
        for i, rel_path in enumerate(self.image_rel_paths):
            if os.path.basename(rel_path) == file_name:
                selected_index = i
                break
        if selected_index >= 0:
            self.switch_image(selected_index)
        else:
            # If not found (perhaps the file isn’t an image), try loading it directly.
            fullpath = os.path.join(directory, file_name)
            try:
                self.pil_image = Image.open(fullpath).convert('RGB')
                self.image_width = self.pil_image.width
                self.image_height = self.pil_image.height
                scale = self.image_size / max(self.pil_image.width, self.pil_image.height)
                new_w = int(self.pil_image.width * scale)
                new_h = int(self.pil_image.height * scale)
                self.pil_image = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                self.set_image_on_label(self.pil_image)
            except Exception:
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # Center Content Setup
    # -----------------------------------------------------------------------
    def setup_content_column(self):
        """
        This sets up the image area, mask editing controls, prompt entry, etc.
        """
        # -- Row 0: top row of buttons/checkbox to set mask mode --
        draw_button = QPushButton("Draw")
        draw_button.setToolTip("Draw a mask using a brush")
        draw_button.clicked.connect(self.draw_mask_editing_mode)
        self.content_layout.addWidget(draw_button, 0, 0)

        fill_button = QPushButton("Fill")
        fill_button.setToolTip("Draw a mask using a fill tool")
        fill_button.clicked.connect(self.fill_mask_editing_mode)
        self.content_layout.addWidget(fill_button, 0, 1)

        self.enable_mask_editing_checkbox = QCheckBox("Enable Mask Editing")
        self.enable_mask_editing_checkbox.setChecked(False)
        self.enable_mask_editing_checkbox.stateChanged.connect(self.set_mask_editing_enabled)
        self.content_layout.addWidget(self.enable_mask_editing_checkbox, 0, 2)

        self.brush_alpha_line = QLineEdit()
        self.brush_alpha_line.setFixedWidth(40)
        self.brush_alpha_line.setText("1.0")
        self.brush_alpha_line.returnPressed.connect(self.update_brush_alpha)
        self.content_layout.addWidget(self.brush_alpha_line, 0, 3)

        alpha_label = QLabel("Brush Alpha")
        self.content_layout.addWidget(alpha_label, 0, 4)

        # Row 1: image display
        self.image_label = ClickableLabel("")
        self.image_label.setFixedSize(self.image_size, self.image_size)
        self.image_label.setMouseTracking(True)
        self.image_label.mouse_moved.connect(self.edit_mask_mouse_move)
        self.image_label.mouse_pressed.connect(self.edit_mask_mouse_press)
        self.image_label.wheel_scrolled.connect(self.draw_mask_radius)
        self.content_layout.addWidget(self.image_label, 1, 0, 1, 5)

        # Row 2: prompt(caption) display and editing box
        self.prompt_line = PlainTextEntry(self.save_current)
        #self.prompt_line.returnPressed.connect(self.save_current)
        self.content_layout.addWidget(self.prompt_line, 2, 0, 1, 5)

    def set_mask_editing_enabled(self, state):
        self.enable_mask_editing = bool(state)

    def update_brush_alpha(self):
        try:
            val = float(self.brush_alpha_line.text())
            self.brush_alpha = max(0.0, min(val, 1.0))
        except ValueError:
            self.brush_alpha = 1.0

    # -----------------------------------------------------------------------
    # Directory scanning and image loading (for next/prev functionality)
    # -----------------------------------------------------------------------
    def load_directory(self, include_subdirectories=False):
        if self.dir and os.path.isdir(self.dir):
            # Optionally, you could update DirectoryBrowser's path here.
            pass
        self.scan_directory(include_subdirectories)
        if len(self.image_rel_paths) > 0:
            self.switch_image(0)
        else:
            self.switch_image(-1)
        self.prompt_line.setFocus()

    def scan_directory(self, include_subdirectories=False):
        def __is_supported_image_extension(filename):
            name, ext = os.path.splitext(filename)
            # Uses your existing helper function to check supported image extensions.
            return path_util.is_supported_image_extension(ext) and not name.endswith("-masklabel")

        self.image_rel_paths.clear()
        if not self.dir or not os.path.isdir(self.dir):
            return

        if include_subdirectories:
            for root, _, files in os.walk(self.dir):
                for filename in files:
                    if __is_supported_image_extension(filename):
                        rel = os.path.relpath(os.path.join(root, filename), self.dir)
                        self.image_rel_paths.append(rel)
        else:
            for filename in os.listdir(self.dir):
                if __is_supported_image_extension(filename):
                    rel = os.path.relpath(os.path.join(self.dir, filename), self.dir)
                    self.image_rel_paths.append(rel)

    # -----------------------------------------------------------------------
    # Image, Mask, Prompt Loading and Switching
    # -----------------------------------------------------------------------
    def load_image(self):
        if self.image_rel_paths and 0 <= self.current_image_index < len(self.image_rel_paths):
            image_rel = self.image_rel_paths[self.current_image_index]
            fullpath = os.path.join(self.dir, image_rel)
            try:
                return Image.open(fullpath).convert('RGB')
            except Exception:
                traceback.print_exc()
                print(f"Could not open image {fullpath}")
        return Image.new("RGB", (512, 512), (0, 0, 0))

    def load_mask(self):
        if self.image_rel_paths and 0 <= self.current_image_index < len(self.image_rel_paths):
            image_rel = self.image_rel_paths[self.current_image_index]
            mask_path = os.path.splitext(image_rel)[0] + "-masklabel.png"
            mask_path = os.path.join(self.dir, mask_path)
            if os.path.exists(mask_path):
                try:
                    return Image.open(mask_path).convert('RGB')
                except Exception:
                    return None
        return None

    # probably should be called "load_caption"
    def load_prompt(self):
        if self.image_rel_paths and 0 <= self.current_image_index < len(self.image_rel_paths):
            image_rel = self.image_rel_paths[self.current_image_index]
            prompt_path = os.path.splitext(image_rel)[0] + ".txt"
            prompt_path = os.path.join(self.dir, prompt_path)
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        return f.read().splitlines()[0]
                except Exception:
                    return ""
        return ""

    # Might be better named "display image and caption"
    def switch_image(self, index: int):
        if index < 0 or index >= len(self.image_rel_paths):
            blank_img = Image.new("RGB", (512, 512), (0, 0, 0))
            self.set_image_on_label(blank_img)
            return

        self.current_image_index = index
        self.pil_image = self.load_image()
        self.pil_mask = self.load_mask()
        self.prompt_text = self.load_prompt()
        self.prompt_line.setPlainText(self.prompt_text)

        if self.pil_image:
            self.image_width = self.pil_image.width
            self.image_height = self.pil_image.height
            scale = self.image_size / max(self.pil_image.width, self.pil_image.height)
            new_w = int(self.pil_image.width * scale)
            new_h = int(self.pil_image.height * scale)
            self.pil_image = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.refresh_image()
        else:
            blank_img = Image.new("RGB", (512, 512), (0, 0, 0))
            self.set_image_on_label(blank_img)

    def refresh_image(self):
        if not self.pil_image:
            return
        if self.pil_mask:
            resized_mask = self.pil_mask.resize(
                (self.pil_image.width, self.pil_image.height),
                Image.Resampling.NEAREST
            )
            if self.display_only_mask:
                self.set_image_on_label(resized_mask)
            else:
                np_image = np.array(self.pil_image).astype(np.float32) / 255.0
                np_mask = np.array(resized_mask).astype(np.float32) / 255.0
                norm_min = 0.3
                mask_min = np_mask.min()
                if mask_min == 0:
                    np_mask = np_mask * (1.0 - norm_min) + norm_min
                elif mask_min < 1:
                    np_mask = ((np_mask - mask_min) / (1.0 - mask_min)) * (1.0 - norm_min) + norm_min
                np_masked_image = (np_image * np_mask * 255.0).astype(np.uint8)
                masked_pil = Image.fromarray(np_masked_image, mode="RGB")
                self.set_image_on_label(masked_pil)
        else:
            self.set_image_on_label(self.pil_image)

    def set_image_on_label(self, pil_image):
        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(
            data, pil_image.width, pil_image.height,
            3 * pil_image.width, QImage.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)

    # -----------------------------------------------------------------------
    # Mouse/Keyboard Handling and Mask Editing
    # -----------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Up:
            self.previous_image()
            return
        elif event.key() == Qt.Key_Down:
            self.next_image()
            return
        elif event.key() == Qt.Key_Return:
            self.save_current()
            return

        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_D:
                self.draw_mask_editing_mode()
                return
            elif event.key() == Qt.Key_F:
                self.fill_mask_editing_mode()
                return
            elif event.key() == Qt.Key_M:
                self.toggle_mask()
                return

        super().keyPressEvent(event)

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.switch_image(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index + 1 < len(self.image_rel_paths):
            self.switch_image(self.current_image_index + 1)

    def edit_mask_mouse_move(self, pos: QPoint):
        if not self.enable_mask_editing:
            return
        self.edit_mask_common(pos, is_press=False)

    def edit_mask_mouse_press(self, pos: QPoint, button: int):
        if not self.enable_mask_editing:
            return
        self.edit_mask_common(pos, is_press=True, button=button)

    def edit_mask_common(self, pos: QPoint, is_press=False, button=None):
        if (
            not self.image_rel_paths or
            self.current_image_index >= len(self.image_rel_paths) or
            self.current_image_index < 0
        ):
            return

        event_x = pos.x()
        event_y = pos.y()
        start_x = int(event_x / self.pil_image.width * self.image_width)
        start_y = int(event_y / self.pil_image.height * self.image_height)
        end_x = int(self.mask_draw_x / self.pil_image.width * self.image_width)
        end_y = int(self.mask_draw_y / self.pil_image.height * self.image_height)

        self.mask_draw_x = event_x
        self.mask_draw_y = event_y

        is_left = (button == Qt.LeftButton)
        is_right = (button == Qt.RightButton)

        if self.mask_editing_mode == 'draw':
            if is_press:
                self.draw_mask(start_x, start_y, end_x, end_y, is_left, is_right)
        elif self.mask_editing_mode == 'fill':
            if is_press:
                self.fill_mask(start_x, start_y, end_x, end_y, is_left, is_right)

    def draw_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None
        if is_left:
            rgb_value = int(self.brush_alpha * 255)
            color = (rgb_value, rgb_value, rgb_value)
        elif is_right:
            color = (0, 0, 0)

        if color is not None:
            if self.pil_mask is None:
                if is_left:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (0, 0, 0))
                else:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (255, 255, 255))

            radius = int(self.mask_draw_radius * max(self.pil_mask.width, self.pil_mask.height))

            draw_obj = ImageDraw.Draw(self.pil_mask)
            draw_obj.line((start_x, start_y, end_x, end_y), fill=color, width=(2 * radius + 1))
            draw_obj.ellipse((start_x - radius, start_y - radius,
                              start_x + radius, start_y + radius), fill=color)
            draw_obj.ellipse((end_x - radius, end_y - radius,
                              end_x + radius, end_y + radius), fill=color)

            self.refresh_image()

    def fill_mask(self, start_x, start_y, end_x, end_y, is_left, is_right):
        color = None
        if is_left:
            rgb_value = int(self.brush_alpha * 255)
            color = (rgb_value, rgb_value, rgb_value)
        elif is_right:
            color = (0, 0, 0)

        if color is not None:
            if self.pil_mask is None:
                if is_left:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (0, 0, 0))
                else:
                    self.pil_mask = Image.new('RGB', (self.image_width, self.image_height), (255, 255, 255))

            np_mask = np.array(self.pil_mask, dtype=np.uint8)
            h, w, _ = np_mask.shape
            if 0 <= start_x < w and 0 <= start_y < h:
                cv2.floodFill(np_mask, None, (start_x, start_y), color)
                self.pil_mask = Image.fromarray(np_mask, 'RGB')
                self.refresh_image()

    def draw_mask_radius(self, delta):
        multiplier = 1.0 + (delta * 0.05)
        self.mask_draw_radius = max(0.0025, self.mask_draw_radius * multiplier)

    def draw_mask_editing_mode(self):
        self.mask_editing_mode = 'draw'

    def fill_mask_editing_mode(self):
        self.mask_editing_mode = 'fill'

    def toggle_mask(self):
        self.display_only_mask = not self.display_only_mask
        self.refresh_image()

    # gets called if Enter pressed on either the image, or the caption for the image
    def save_current(self):
        if (
            not self.image_rel_paths or
            self.current_image_index >= len(self.image_rel_paths) or
            self.current_image_index < 0
        ):
            return
        image_rel = self.image_rel_paths[self.current_image_index]

        prompt_path = os.path.splitext(image_rel)[0] + ".txt"
        prompt_path = os.path.join(self.dir, prompt_path)
        mask_path = os.path.splitext(image_rel)[0] + "-masklabel.png"
        mask_path = os.path.join(self.dir, mask_path)

        self.prompt_text = self.prompt_line.toPlainText()
        try:
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(self.prompt_text)
        except Exception:
            traceback.print_exc()

        if self.pil_mask:
            try:
                self.pil_mask.save(mask_path)
            except Exception:
                traceback.print_exc()

    def open_directory(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select Directory", self.dir or "")
        if new_dir:
            self.dir = new_dir
            self.load_directory(self.config_ui_data["include_subdirectories"])

    def open_mask_window(self):
        dialog = GenerateMasksWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
        dialog.exec()
        self.switch_image(self.current_image_index)

    def open_caption_window(self):
        dialog = GenerateCaptionsWindow(self, self.dir, self.config_ui_data["include_subdirectories"])
        if hasattr(dialog, "exec"):
            dialog.exec()
        else:
            dialog.show()
        self.switch_image(self.current_image_index)

    def open_in_explorer(self):
        try:
            if (
                not self.image_rel_paths or
                self.current_image_index >= len(self.image_rel_paths) or
                self.current_image_index < 0
            ):
                return
            image_rel = self.image_rel_paths[self.current_image_index]
            fullpath = os.path.realpath(os.path.join(self.dir, image_rel))
            subprocess.Popen(f"explorer /select,{fullpath}")
        except Exception:
            traceback.print_exc()

    def load_masking_model(self, model):
        self.captioning_model = None

        if model == "ClipSeg":
            if not isinstance(self.masking_model, ClipSegModel):
                print("loading ClipSeg model, this may take a while")
                self.masking_model = ClipSegModel(default_device, torch.float32)
        elif model == "Rembg":
            if not isinstance(self.masking_model, RembgModel):
                print("loading Rembg model, this may take a while")
                self.masking_model = RembgModel(default_device, torch.float32)
        elif model == "Rembg-Human":
            if not isinstance(self.masking_model, RembgHumanModel):
                print("loading Rembg-Human model, this may take a while")
                self.masking_model = RembgHumanModel(default_device, torch.float32)
        elif model == "Hex Color":
            if not isinstance(self.masking_model, MaskByColor):
                self.masking_model = MaskByColor(default_device, torch.float32)

    def print_help(self):
        QMessageBox.information(self, "Help", self.help_text)

# -------------------------------------------------------------------------
# Helper: ClickableLabel class 
# -------------------------------------------------------------------------
from PySide6.QtCore import Signal

class ClickableLabel(QLabel):
    """
    A QLabel that can emit a clicked signal and also
    let us track mouseMove and wheel events.
    """
    clicked_index = Signal(int)
    mouse_moved = Signal(QPoint)
    mouse_pressed = Signal(QPoint, int)
    wheel_scrolled = Signal(float)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.index = -1
        self.setCursor(Qt.ArrowCursor)

    def set_index(self, index: int):
        self.index = index

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            if self.index >= 0:
                self.clicked_index.emit(self.index)
            self.mouse_pressed.emit(event.pos(), event.button())

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        self.mouse_moved.emit(event.pos())

    def wheelEvent(self, event):
        delta_y = event.angleDelta().y()
        direction = 1 if delta_y > 0 else -1
        self.wheel_scrolled.emit(direction)
        event.accept()


# ----------------------------------------------------
# Subclass of QPlainTextEdit
#  acts a little like QLineEntry, in that it triggers from ENTER key
# ----------------------------------------------------

class PlainTextEntry(QPlainTextEdit):
    def __init__(self, on_enter_pressed, linecount=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_enter_pressed = on_enter_pressed
        # defaults to 3 lines. lets make it 2 instead
        font_metrics = self.fontMetrics()
        line_height = font_metrics.lineSpacing()
        self.setMinimumHeight(line_height * linecount + 10)

    def keyPressEvent(self, event):
        # Special case ENTER, but pass through everything else
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() == Qt.NoModifier:
                if callable(self.on_enter_pressed):
                    self.on_enter_pressed()
                return  # prevent newline character
        super().keyPressEvent(event)
