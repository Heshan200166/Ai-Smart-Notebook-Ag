"""
Drawing Engine Module
======================
Manages the virtual canvas, drawing operations, color/brush controls,
undo history, and canvas-to-frame overlay blending.
"""

import cv2
import numpy as np
import os
from datetime import datetime


class DrawingEngine:
    """Virtual canvas with drawing, erasing, undo, and overlay capabilities."""

    # Predefined color palette (BGR format for OpenCV)
    COLORS = {
        "Cyan":    (255, 255, 0),
        "Red":     (0, 0, 255),
        "Green":   (0, 255, 0),
        "Blue":    (255, 0, 0),
        "Yellow":  (0, 255, 255),
        "Magenta": (255, 0, 255),
        "White":   (255, 255, 255),
        "Orange":  (0, 165, 255),
    }

    COLOR_LIST = list(COLORS.values())
    COLOR_NAMES = list(COLORS.keys())

    # Brush size presets
    BRUSH_SIZES = [3, 5, 8, 12, 18, 25]

    def __init__(self, width=1280, height=720):
        """
        Initialize the drawing engine.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self.width = width
        self.height = height

        # The drawing canvas (black background)
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Current drawing settings
        self.color = self.COLOR_LIST[0]  # Cyan
        self.color_index = 0
        self.brush_size = 5
        self.brush_size_index = 1
        self.eraser_mode = False

        # Previous point for line smoothing
        self._prev_point = None

        # Undo history (stores canvas snapshots)
        self._undo_stack = []
        self._max_undo = 20

        # Jump threshold: max distance between consecutive points before breaking the line
        self._jump_threshold = 80

    def draw(self, x, y):
        """
        Draw at the given position. Connects to previous point for smooth lines.

        Args:
            x: X coordinate on canvas.
            y: Y coordinate on canvas.
        """
        draw_color = (0, 0, 0) if self.eraser_mode else self.color
        thickness = self.brush_size * 3 if self.eraser_mode else self.brush_size

        if self._prev_point is not None:
            # Calculate distance to prevent jumps
            dist = np.sqrt(
                (x - self._prev_point[0]) ** 2 +
                (y - self._prev_point[1]) ** 2
            )

            if dist < self._jump_threshold:
                # Draw a line from previous to current point
                cv2.line(
                    self.canvas,
                    self._prev_point, (x, y),
                    draw_color, thickness,
                    lineType=cv2.LINE_AA
                )

        # Draw a circle at the current point (round cap)
        cv2.circle(self.canvas, (x, y), thickness // 2, draw_color, -1, lineType=cv2.LINE_AA)

        self._prev_point = (x, y)

    def stop_drawing(self):
        """Stop the current stroke (reset previous point)."""
        if self._prev_point is not None:
            self._save_undo_state()
        self._prev_point = None

    def set_color(self, color_index):
        """
        Set brush color by index.

        Args:
            color_index: Index into COLOR_LIST.
        """
        if 0 <= color_index < len(self.COLOR_LIST):
            self.color_index = color_index
            self.color = self.COLOR_LIST[color_index]
            self.eraser_mode = False

    def set_brush_size(self, size_index):
        """
        Set brush size by index.

        Args:
            size_index: Index into BRUSH_SIZES.
        """
        if 0 <= size_index < len(self.BRUSH_SIZES):
            self.brush_size_index = size_index
            self.brush_size = self.BRUSH_SIZES[size_index]

    def toggle_eraser(self):
        """Toggle eraser mode on/off."""
        self.eraser_mode = not self.eraser_mode

    def set_eraser(self, enabled):
        """Explicitly set eraser mode."""
        self.eraser_mode = enabled

    def clear_canvas(self):
        """Clear the entire canvas."""
        self._save_undo_state()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._prev_point = None

    def undo(self):
        """Undo the last drawing action."""
        if self._undo_stack:
            self.canvas = self._undo_stack.pop()
            self._prev_point = None

    def save_canvas(self, save_dir="sessions"):
        """
        Save the current canvas as a PNG image.

        Args:
            save_dir: Directory to save into.

        Returns:
            The filepath of the saved image.
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, self.canvas)
        return filepath

    def get_overlay(self, frame):
        """
        Overlay the canvas onto the camera frame using additive blending.

        Args:
            frame: BGR camera frame (must match canvas dimensions).

        Returns:
            Blended frame with canvas overlay.
        """
        # Resize canvas to match frame if needed
        if frame.shape[:2] != (self.height, self.width):
            canvas_resized = cv2.resize(self.canvas, (frame.shape[1], frame.shape[0]))
        else:
            canvas_resized = self.canvas

        # Create mask where canvas has content
        gray = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Darken the frame where we'll draw, then add canvas
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas_resized, canvas_resized, mask=mask)

        return cv2.add(frame_bg, canvas_fg)

    def has_content(self):
        """Check if the canvas has any drawing content."""
        return np.any(self.canvas > 0)

    def _save_undo_state(self):
        """Save current canvas state to undo stack."""
        if len(self._undo_stack) >= self._max_undo:
            self._undo_stack.pop(0)
        self._undo_stack.append(self.canvas.copy())
