"""
Text Recognizer Module (EasyOCR)
==================================
Extracts handwritten or printed text from canvas regions using EasyOCR.
Includes image preprocessing optimized for air-drawn text on dark backgrounds.
"""

import cv2
import numpy as np
import threading


class TextRecognizer:
    """OCR engine using EasyOCR for text extraction from canvas regions."""

    def __init__(self, languages=None):
        """
        Initialize the text recognizer.

        Args:
            languages: List of language codes (default: ['en']).
        """
        self._languages = languages or ['en']
        self._reader = None
        self._lock = threading.Lock()
        self._initialized = False
        self._init_error = None

    def _ensure_initialized(self):
        """Lazy-initialize EasyOCR reader (heavy — only on first use)."""
        if self._initialized:
            return self._init_error is None

        with self._lock:
            if self._initialized:
                return self._init_error is None

            try:
                import easyocr
                self._reader = easyocr.Reader(
                    self._languages,
                    gpu=False,      # CPU mode for compatibility
                    verbose=False
                )
                self._initialized = True
                return True
            except Exception as e:
                self._init_error = str(e)
                self._initialized = True
                return False

    def recognize(self, image: np.ndarray) -> str:
        """
        Extract text from a canvas image region.

        Args:
            image: BGR numpy array (cropped region of the canvas).

        Returns:
            Extracted text string, or an error message.
        """
        if not self._ensure_initialized():
            return f"[OCR Error: {self._init_error}]"

        if image is None or image.size == 0:
            return "[No image data]"

        # Preprocess for better OCR accuracy
        processed = self.preprocess(image)

        try:
            # Run EasyOCR
            results = self._reader.readtext(
                processed,
                detail=0,              # Return text only (no bounding boxes)
                paragraph=True,        # Group text into paragraphs
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
            )

            if not results:
                return "[No text detected]"

            return " ".join(results).strip()

        except Exception as e:
            return f"[OCR Error: {e}]"

    def recognize_detailed(self, image: np.ndarray) -> list:
        """
        Extract text with bounding boxes and confidence scores.

        Args:
            image: BGR numpy array.

        Returns:
            List of (bbox, text, confidence) tuples.
        """
        if not self._ensure_initialized():
            return []

        if image is None or image.size == 0:
            return []

        processed = self.preprocess(image)

        try:
            results = self._reader.readtext(
                processed,
                detail=1,
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
            )
            return results
        except Exception:
            return []

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess canvas crop for optimal OCR accuracy.

        The canvas has colored ink on a black background, which is
        the opposite of what OCR expects (dark text on white).
        This method inverts and enhances the image.

        Args:
            image: BGR canvas crop.

        Returns:
            Processed grayscale image ready for OCR.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Invert: black background → white background, colored ink → dark ink
        inverted = cv2.bitwise_not(gray)

        # Apply slight Gaussian blur to smooth jagged air-drawn strokes
        blurred = cv2.GaussianBlur(inverted, (3, 3), 0)

        # Adaptive thresholding for clean binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Dilate slightly to connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Add white border padding (helps OCR with edge text)
        padded = cv2.copyMakeBorder(
            dilated, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=255
        )

        return padded

    def is_available(self) -> bool:
        """Check if EasyOCR is available and can be initialized."""
        return self._ensure_initialized()
