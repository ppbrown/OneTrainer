from util.import_util import script_imports
import signal

script_imports()

from modules.ui.CaptionUI import CaptionUI
from modules.util.args.CaptionUIArgs import CaptionUIArgs
from modules.util.ui.ui_utils import set_window_icon


import sys
from PySide6.QtWidgets import QApplication

def handle_sigint(*args):
    # Note that it takes a few seconds for this to be called
    print("Interrupt caught: Ctrl+C pressed, exiting.")
    QApplication.quit()

def main():
    # signal.signal(signal.SIGINT, handle_sigint)
    app = QApplication(sys.argv)
    set_window_icon(app)


    window = CaptionUI()
    window.showNormal()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
