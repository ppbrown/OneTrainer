from util.import_util import script_imports

script_imports()

from modules.ui.TrainUI import TrainUI
from modules.util.ui.ui_utils import set_window_icon

import sys
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    set_window_icon(app)
    # in theory thats all we would need to do...
    # Qt makes that one call apply for ALL related windows.
    # Except that MS-windows taskbar is stupid and wont show the icon, unless the app
    # has been converted with pyinstaller

    window = TrainUI()
    window.setWindowTitle("OneTrainer")
    window.showNormal()

    sys.exit(app.exec())
    

if __name__ == '__main__':
    main()
