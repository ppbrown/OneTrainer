

import sys
from collections.abc import Callable
from typing import Any, Optional
from pathlib import Path

from PySide6.QtCore import QObject, QEvent
from PySide6.QtGui import QWheelEvent, QIcon
from PySide6.QtWidgets import QWidget

"""
Usage:
def on_wheel(delta: int, event: QWheelEvent):
    print("Scrolled:", "up" if delta > 0 else "down")

bind_mousewheel(myWidget, {myWidget}, on_wheel)
"""
def bind_mousewheel(
    widget: QWidget,
    whitelist: Optional[set[QWidget]],
    callback: Callable[[int, QWheelEvent], None]
):

    class MouseWheelFilter(QObject):
        def eventFilter(self, obj: QObject, event: QEvent) -> bool:
            if event.type() == QEvent.Type.Wheel and isinstance(event, QWheelEvent):
                if whitelist is not None and obj not in whitelist:
                    return False
                # angleDelta().y() is positive => scroll up, negative => scroll down
                delta_y = event.angleDelta().y()
                sign = 1 if delta_y > 0 else -1
                callback(sign, event)
                return True
            return super().eventFilter(obj, event)

    mw_filter = MouseWheelFilter(widget)
    widget.installEventFilter(mw_filter)


def set_window_icon(window: QWidget) -> None:
    """Set the application window icon based on the current platform

    Args:
        window: The window object to set the icon for
    """

    icon_dir = Path("resources/icons")
    img_path = icon_dir / "icon.png"

    # Through the magic of Qt, this should theoretically work for all platforms
    try:
        icon = QIcon(str(img_path))
        window.setWindowIcon(icon)

    except Exception as e:
        print(f"Failed to set window icon: {e}")
