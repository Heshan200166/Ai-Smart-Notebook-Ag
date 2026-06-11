"""
Drawing Engine Module
======================
Manages the virtual canvas, drawing operations, color/brush controls,
undo history, and canvas-to-frame overlay blending.

Enhanced with:
- Bézier curve interpolation for ultra-smooth lines
- Point buffer for natural-looking strokes
- Adaptive jump threshold based on brush size
- Cursor position smoothing (separate from landmark smoothing)
"""

import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque


class DrawingEngine:
    """Virtual canvas with smooth drawing, erasing, undo, and overlay capabilities."""

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

        # --- Smoothing enhancements ---

        # Cursor position smoothing (EMA on draw coordinates)
        self._smooth_x = None
        self._smooth_y = None
        self._cursor_smoothing = 0.45  # Lower = smoother, higher = more responsive

        # Point buffer for Bézier curve interpolation
        self._point_buffer = deque(maxlen=4)

        # Adaptive jump threshold (scales with brush size)
        self._base_jump_threshold = 100

        # Track if a stroke has been started (for undo grouping)
        self._stroke_active = False

    @property
    def _jump_threshold(self):
        """Adaptive jump threshold based on brush size."""
        return self._base_jump_threshold + self.brush_size * 2

    def draw(self, x, y):
        """
        Draw at the given position with smoothing and interpolation.

        Args:
            x: X coordinate on canvas.
            y: Y coordinate on canvas.
        """
        # Apply cursor position smoothing
        if self._smooth_x is None:
            self._smooth_x = float(x)
            self._smooth_y = float(y)
        else:
            alpha = self._cursor_smoothing
            self._smooth_x = alpha * x + (1 - alpha) * self._smooth_x
            self._smooth_y = alpha * y + (1 - alpha) * self._smooth_y

        sx, sy = int(self._smooth_x), int(self._smooth_y)

        # Save undo state at stroke start
        if not self._stroke_active:
            self._save_undo_state()
            self._stroke_active = True

        draw_color = (0, 0, 0) if self.eraser_mode else self.color
        thickness = self.brush_size * 3 if self.eraser_mode else self.brush_size

        # Add to point buffer
        self._point_buffer.append((sx, sy))

        if self._prev_point is not None:
            dist = np.sqrt(
                (sx - self._prev_point[0]) ** 2 +
                (sy - self._prev_point[1]) ** 2
            )

            if dist < self._jump_threshold:
                if len(self._point_buffer) >= 3:
                    # Use quadratic Bézier interpolation for smooth curves
                    self._draw_smooth_line(draw_color, thickness)
                else:
                    # Not enough points yet; draw simple line
                    cv2.line(
                        self.canvas,
                        self._prev_point, (sx, sy),
                        draw_color, thickness,
                        lineType=cv2.LINE_AA
                    )
            # If distance exceeds threshold, skip (prevents wild jumps)

        # Draw circle at current point (round cap)
        cv2.circle(self.canvas, (sx, sy), thickness // 2, draw_color, -1, lineType=cv2.LINE_AA)

        self._prev_point = (sx, sy)

    def _draw_smooth_line(self, color, thickness):
        """
        Draw a smooth line using quadratic Bézier interpolation
        through recent points in the buffer.
        """
        points = list(self._point_buffer)
        n = len(points)

        if n < 3:
            # Fallback to simple line
            if n == 2:
                cv2.line(self.canvas, points[0], points[1], color, thickness, cv2.LINE_AA)
            return

        # Use last 3 points for quadratic Bézier
        p0 = np.array(points[-3], dtype=np.float64)
        p1 = np.array(points[-2], dtype=np.float64)
        p2 = np.array(points[-1], dtype=np.float64)

        # Generate interpolated points along the curve
        num_steps = max(int(np.linalg.norm(p2 - p0) / 3), 4)

        prev_pt = None
        for i in range(num_steps + 1):
            t = i / num_steps
            # Quadratic Bézier: B(t) = (1-t)²·P0 + 2(1-t)t·P1 + t²·P2
            pt = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
            pt_int = (int(pt[0]), int(pt[1]))

            if prev_pt is not None:
                cv2.line(self.canvas, prev_pt, pt_int, color, thickness, cv2.LINE_AA)

            prev_pt = pt_int

    def stop_drawing(self):
        """Stop the current stroke (reset previous point and smoothing)."""
        self._prev_point = None
        self._smooth_x = None
        self._smooth_y = None
        self._point_buffer.clear()
        self._stroke_active = False

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
        self._smooth_x = None
        self._smooth_y = None
        self._point_buffer.clear()
        self._stroke_active = False

    def undo(self):
        """Undo the last drawing action."""
        if self._undo_stack:
            self.canvas = self._undo_stack.pop()
            self._prev_point = None
            self._smooth_x = None
            self._smooth_y = None
            self._point_buffer.clear()
            self._stroke_active = False

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
        Overlay the canvas onto the camera frame using masking.

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
