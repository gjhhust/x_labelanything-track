"""This module defines the main application window"""

from PyQt5.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from ..app_info import __appdescription__, __appname__
from .labeling.label_wrapper import LabelingWrapper
import sys
import logging
from anylabeling.views.labeling.logger import logger
import traceback

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(
        self,
        app,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        super().__init__()
        self.app = app
        self.config = config

        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle(__appname__)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.labeling_widget = LabelingWrapper(
            self,
            config=config,
            filename=filename,
            output=output,
            output_file=output_file,
            output_dir=output_dir,
        )
        main_layout.addWidget(self.labeling_widget)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        status_bar = QStatusBar()
        status_bar.showMessage(f"{__appname__} - {__appdescription__}")
        self.setStatusBar(status_bar)

def excepthook(type, value, traceback_obj):
    error_message = f"An unhandled exception occurred:\n\n{type.__name__}: {value}"

    # Get the traceback as a string
    traceback_str = "".join(traceback.format_exception(type, value, traceback_obj))

    # Save the error information to a file (you can customize the file path)
    with open("error.log", "a") as error_file:
        error_file.write(error_message + "\n" + traceback_str + "\n\n\n\n")

    sys.exit(0)

# Install the custom excepthook
sys.excepthook = excepthook