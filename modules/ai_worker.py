"""
AI Worker Module
==================
QThread-based background worker for AI operations.
Runs OCR, math solving, and shape recognition asynchronously
so the camera feed stays smooth at 30+ FPS.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np

from modules.text_recognizer import TextRecognizer
from modules.math_solver import MathSolver, MathResult
from modules.shape_recognizer import ShapeRecognizer, ShapeResult


class AIWorker(QThread):
    """Background thread for AI processing tasks."""

    # Signals emitted when processing completes
    text_recognized = pyqtSignal(str)           # OCR result text
    math_solved = pyqtSignal(object)            # MathResult object
    shape_detected = pyqtSignal(object)         # ShapeResult object
    status_message = pyqtSignal(str)            # Status updates
    error_occurred = pyqtSignal(str)            # Error messages

    def __init__(self, parent=None):
        super().__init__(parent)

        # AI modules (initialized lazily where needed)
        self._text_recognizer = TextRecognizer()
        self._math_solver = MathSolver()
        self._shape_recognizer = ShapeRecognizer()

        # Task queue (simple — one task at a time)
        self._task = None
        self._task_data = None
        self._running = False

    def process_ocr(self, image: np.ndarray):
        """
        Queue an OCR task.

        Args:
            image: BGR numpy array of the canvas region to read.
        """
        self._task = "ocr"
        self._task_data = image.copy()
        if not self.isRunning():
            self.start()

    def process_math(self, text: str):
        """
        Queue a math solving task.

        Args:
            text: Mathematical expression text to evaluate.
        """
        self._task = "math"
        self._task_data = text
        if not self.isRunning():
            self.start()

    def process_shape(self, points: list):
        """
        Queue a shape recognition task.

        Args:
            points: List of (x, y) stroke points.
        """
        self._task = "shape"
        self._task_data = points
        if not self.isRunning():
            self.start()

    def process_ocr_and_math(self, image: np.ndarray):
        """
        Queue a combined OCR + Math task.
        First extracts text, then tries to solve it as math.

        Args:
            image: BGR numpy array of the canvas region.
        """
        self._task = "ocr_math"
        self._task_data = image.copy()
        if not self.isRunning():
            self.start()

    def run(self):
        """Execute the queued task."""
        self._running = True

        try:
            if self._task == "ocr":
                self._run_ocr(self._task_data)

            elif self._task == "math":
                self._run_math(self._task_data)

            elif self._task == "shape":
                self._run_shape(self._task_data)

            elif self._task == "ocr_math":
                self._run_ocr_math(self._task_data)

        except Exception as e:
            self.error_occurred.emit(f"AI Worker error: {str(e)}")

        finally:
            self._running = False
            self._task = None
            self._task_data = None

    def _run_ocr(self, image):
        """Execute OCR processing."""
        self.status_message.emit("🔍 Reading text...")

        text = self._text_recognizer.recognize(image)
        self.text_recognized.emit(text)

        if text and not text.startswith("["):
            self.status_message.emit(f"✅ Text: \"{text[:50]}...\"" if len(text) > 50 else f"✅ Text: \"{text}\"")
        else:
            self.status_message.emit("⚠️ No text detected")

    def _run_math(self, text):
        """Execute math solving."""
        self.status_message.emit("🧮 Solving math...")

        result = self._math_solver.solve(text)
        self.math_solved.emit(result)

        if result.success:
            self.status_message.emit(f"✅ Result: {result.solution}")
        else:
            self.status_message.emit(f"⚠️ {result.error}")

    def _run_shape(self, points):
        """Execute shape recognition."""
        result = self._shape_recognizer.analyze_stroke(points)
        self.shape_detected.emit(result)

        if result.shape_type != "unknown":
            self.status_message.emit(
                f"📐 Detected: {result.shape_type} ({result.confidence:.0%})"
            )

    def _run_ocr_math(self, image):
        """Execute OCR followed by math solving."""
        self.status_message.emit("🔍 Reading text...")

        text = self._text_recognizer.recognize(image)
        self.text_recognized.emit(text)

        if text and not text.startswith("["):
            self.status_message.emit(f"📝 Text: \"{text}\"")

            # Check if it looks like math
            if self._math_solver.is_math_expression(text):
                self.status_message.emit("🧮 Solving math...")
                result = self._math_solver.solve(text)
                self.math_solved.emit(result)

                if result.success:
                    self.status_message.emit(f"✅ {text} = {result.solution}")
                else:
                    self.status_message.emit(f"⚠️ {result.error}")
            else:
                self.status_message.emit("📝 Text extracted (not a math expression)")
        else:
            self.status_message.emit("⚠️ No text detected")

    def is_busy(self) -> bool:
        """Check if the worker is currently processing a task."""
        return self._running
