"""
Canvas Widget
==============
Custom QWidget that displays OpenCV frames as QImages.
Handles BGR→RGB conversion, aspect-ratio-preserving scaling, and mouse events.
"""

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtCore import Qt, QSize
import numpy as np


class CanvasWidget(QWidget):
    """Widget for displaying the OpenCV camera feed + canvas overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #0a0a0f; border-radius: 12px;")

    def update_frame(self, frame):
        """
        Update the displayed frame.

        Args:
            frame: BGR numpy array from OpenCV.
        """
        if frame is None:
            return

        # Convert BGR to RGB
        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_img = QImage(
            rgb_frame.data,
            w, h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        self._pixmap = QPixmap.fromImage(q_img)
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Paint the current frame scaled to fit the widget."""
        if self._pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Scale pixmap to fit widget while preserving aspect ratio
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Center the scaled pixmap
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()

    def sizeHint(self):
        return QSize(1280, 720)
