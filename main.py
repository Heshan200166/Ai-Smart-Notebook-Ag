"""
AI Smart Air Notebook
======================
Main entry point for the application.

An intelligent touchless note-taking system that enables users to write,
draw, and interact with digital content using hand gestures.

Usage:
    python main.py
"""

import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    """Launch the AI Smart Air Notebook application."""
    # Enable High-DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("AI Smart Air Notebook")
    app.setOrganizationName("AI Smart Notebook")

    # Set default font
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
