
import os
import random
import sys
import threads
from os.path import join, abspath
from main_gui import QtCore, QtWidgets

def transform_to_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path (str): path to the file.

    Returns:
        str: path that works for both debug and standalone app
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = abspath(".")

    return join(base_path, relative_path)


# displaying a message
def display_message(icon, main_message, informative_message, window_title):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(icon)
    msg.setText(main_message)
    msg.setInformativeText(informative_message)
    msg.setWindowTitle(window_title)
    msg.exec()

def get_project_root_directory():
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    root_directory = path + '/'
    return root_directory
