"""
Shape Recognizer Module
========================
Analyzes completed drawing strokes and detects geometric shapes.
When a shape is detected with high confidence, the rough stroke can be
replaced with a perfect geometric shape.

Supported shapes:
- Lines (straight strokes)
- Circles / Ellipses
- Rectangles / Squares
- Triangles
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ShapeResult:
    """Result of shape analysis."""
    shape_type: str         # "line", "circle", "rectangle", "triangle", "unknown"
    confidence: float       # 0.0 to 1.0
    parameters: dict        # Shape-specific parameters
    original_points: list   # The original stroke points


class ShapeRecognizer:
    """Detects and renders perfect geometric shapes from rough strokes."""

    # Minimum confidence to accept a shape detection
    MIN_CONFIDENCE = 0.70

    # Minimum number of points in a stroke to analyze
    MIN_POINTS = 8

    def analyze_stroke(self, points: List[Tuple[int, int]]) -> ShapeResult:
        """
        Analyze a completed stroke and detect if it forms a geometric shape.

        Args:
            points: List of (x, y) coordinates from the stroke.

        Returns:
            ShapeResult with detected shape type, confidence, and parameters.
        """
        if len(points) < self.MIN_POINTS:
            return ShapeResult("unknown", 0.0, {}, points)

        pts = np.array(points, dtype=np.float32)

        # Try each shape detector, pick the best match
        candidates = [
            self._detect_line(pts),
            self._detect_circle(pts),
            self._detect_polygon(pts),
        ]

        # Filter valid candidates and pick highest confidence
        valid = [c for c in candidates if c.confidence >= self.MIN_CONFIDENCE]

        if not valid:
            return ShapeResult("unknown", 0.0, {}, points)

        return max(valid, key=lambda c: c.confidence)

    def render_shape(self, canvas: np.ndarray, result: ShapeResult,
                     color: tuple, thickness: int) -> np.ndarray:
        """
        Draw a perfect geometric shape onto the canvas.

        Args:
            canvas: The drawing canvas (numpy array).
            result: ShapeResult from analyze_stroke.
            color: BGR color tuple.
            thickness: Line thickness.

        Returns:
            The canvas with the perfect shape drawn.
        """
        if result.shape_type == "line":
            p1 = result.parameters["start"]
            p2 = result.parameters["end"]
            cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

        elif result.shape_type == "circle":
            center = result.parameters["center"]
            radius = result.parameters["radius"]
            cv2.circle(canvas, center, radius, color, thickness, cv2.LINE_AA)

        elif result.shape_type == "rectangle":
            vertices = result.parameters["vertices"]
            for i in range(4):
                cv2.line(canvas, vertices[i], vertices[(i + 1) % 4],
                         color, thickness, cv2.LINE_AA)

        elif result.shape_type == "triangle":
            vertices = result.parameters["vertices"]
            for i in range(3):
                cv2.line(canvas, vertices[i], vertices[(i + 1) % 3],
                         color, thickness, cv2.LINE_AA)

        return canvas

    def _detect_line(self, pts: np.ndarray) -> ShapeResult:
        """Detect a straight line (start-to-end distance vs total arc length)."""
        start = pts[0]
        end = pts[-1]

        # Direct distance
        direct_dist = np.linalg.norm(end - start)

        # Arc length (sum of distances between consecutive points)
        diffs = np.diff(pts, axis=0)
        arc_length = np.sum(np.linalg.norm(diffs, axis=1))

        if arc_length < 1e-6:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        # Straightness ratio (1.0 = perfectly straight)
        straightness = direct_dist / arc_length

        # Also check if it's long enough to be meaningful
        if direct_dist < 30:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        # Confidence mapping
        if straightness > 0.92:
            confidence = 0.95
        elif straightness > 0.85:
            confidence = 0.80
        else:
            confidence = straightness * 0.8

        return ShapeResult(
            "line", confidence,
            {
                "start": (int(start[0]), int(start[1])),
                "end": (int(end[0]), int(end[1]))
            },
            pts.tolist()
        )

    def _detect_circle(self, pts: np.ndarray) -> ShapeResult:
        """Detect a circle using minimum enclosing circle and circularity."""
        pts_int = pts.astype(np.int32)

        # Need to create a contour-like structure
        contour = pts_int.reshape(-1, 1, 2)

        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < 15:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        # Calculate how well points fit the circle
        distances = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Coefficient of variation (lower = more circular)
        cv = std_dist / (mean_dist + 1e-6)

        # Check if the stroke is closed (start near end)
        closure_dist = np.linalg.norm(pts[0] - pts[-1])
        closure_ratio = closure_dist / (2 * radius + 1e-6)
        is_closed = closure_ratio < 0.4

        if not is_closed:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        # Confidence from circularity
        if cv < 0.08:
            confidence = 0.95
        elif cv < 0.15:
            confidence = 0.85
        elif cv < 0.25:
            confidence = 0.72
        else:
            confidence = 0.0

        return ShapeResult(
            "circle", confidence,
            {
                "center": (int(cx), int(cy)),
                "radius": int(radius)
            },
            pts.tolist()
        )

    def _detect_polygon(self, pts: np.ndarray) -> ShapeResult:
        """Detect triangles and rectangles using contour approximation."""
        pts_int = pts.astype(np.int32)
        contour = pts_int.reshape(-1, 1, 2)

        # Check if stroke is closed
        closure_dist = np.linalg.norm(pts[0] - pts[-1])
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 1e-6:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        closure_ratio = closure_dist / (perimeter + 1e-6)
        if closure_ratio > 0.25:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        # Close the contour by appending the first point
        closed_contour = np.vstack([contour, contour[0:1]])

        # Try different epsilon values for approximation
        best_result = ShapeResult("unknown", 0.0, {}, pts.tolist())

        for eps_factor in [0.02, 0.03, 0.04, 0.05]:
            epsilon = eps_factor * perimeter
            approx = cv2.approxPolyDP(closed_contour, epsilon, True)
            n_vertices = len(approx)

            if n_vertices == 3:
                result = self._validate_triangle(approx, perimeter, pts)
                if result.confidence > best_result.confidence:
                    best_result = result

            elif n_vertices == 4:
                result = self._validate_rectangle(approx, perimeter, pts)
                if result.confidence > best_result.confidence:
                    best_result = result

        return best_result

    def _validate_triangle(self, approx, perimeter, pts) -> ShapeResult:
        """Validate a 3-vertex approximation as a triangle."""
        vertices = approx.reshape(-1, 2)

        # Check that all sides have reasonable length
        sides = []
        for i in range(3):
            side_len = np.linalg.norm(vertices[i] - vertices[(i + 1) % 3])
            sides.append(side_len)

        min_side = min(sides)
        max_side = max(sides)

        if min_side < 20:
            return ShapeResult("unknown", 0.0, {}, pts.tolist())

        # Side ratio check (no side should be too small compared to others)
        side_ratio = min_side / (max_side + 1e-6)

        if side_ratio > 0.3:
            confidence = 0.85
        elif side_ratio > 0.15:
            confidence = 0.72
        else:
            confidence = 0.0

        verts = [(int(v[0]), int(v[1])) for v in vertices]

        return ShapeResult(
            "triangle", confidence,
            {"vertices": verts},
            pts.tolist()
        )

    def _validate_rectangle(self, approx, perimeter, pts) -> ShapeResult:
        """Validate a 4-vertex approximation as a rectangle."""
        vertices = approx.reshape(-1, 2)

        # Check angles — all should be close to 90°
        angles = []
        for i in range(4):
            v1 = vertices[(i - 1) % 4] - vertices[i]
            v2 = vertices[(i + 1) % 4] - vertices[i]

            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        # Average deviation from 90°
        angle_deviation = np.mean([abs(a - 90) for a in angles])

        if angle_deviation < 10:
            confidence = 0.92
        elif angle_deviation < 20:
            confidence = 0.78
        elif angle_deviation < 30:
            confidence = 0.65
        else:
            confidence = 0.0

        verts = [(int(v[0]), int(v[1])) for v in vertices]

        return ShapeResult(
            "rectangle", confidence,
            {"vertices": verts},
            pts.tolist()
        )
