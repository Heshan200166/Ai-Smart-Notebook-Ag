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
        This method thresholds the digital ink, dilates it to connect and
        bolden strokes, blurs/re-thresholds for smooth edges, and inverts it.

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

        # Threshold to get white ink (255) on black background (0).
        # Threshold at 2 to capture even very faint drawing stroke pixels.
        _, binary = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)

        # Dilate the white ink to connect strokes and make them thicker/clearer
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Apply a slight Gaussian blur to smooth the edges of the strokes
        blurred = cv2.GaussianBlur(dilated, (3, 3), 0)

        # Re-threshold to get sharp edges
        _, sharp = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Invert to get black ink (0) on white background (255)
        inverted = cv2.bitwise_not(sharp)

        # Add white border padding (helps EasyOCR detect text near borders)
        padded = cv2.copyMakeBorder(
            inverted, 30, 30, 30, 30,
            cv2.BORDER_CONSTANT, value=255
        )

        return padded

    def is_available(self) -> bool:
        """Check if EasyOCR is available and can be initialized."""
        return self._ensure_initialized()
