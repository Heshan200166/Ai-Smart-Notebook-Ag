"""
Hand Tracker Module
====================
Wraps MediaPipe HandLandmarker (Tasks API) for real-time hand landmark
detection and finger state analysis.
Provides 21 landmark positions and finger up/down states for gesture recognition.
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)


class HandTracker:
    """Real-time hand tracking using MediaPipe HandLandmarker Tasks API."""

    # MediaPipe landmark IDs for fingertips and their corresponding joints
    FINGERTIP_IDS = [4, 8, 12, 16, 20]       # Thumb, Index, Middle, Ring, Pinky tips
    FINGER_PIP_IDS = [2, 6, 10, 14, 18]      # Corresponding PIP/IP joints

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Connections for drawing landmarks
    HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        """
        Initialize the hand tracker.

        Args:
            max_hands: Maximum number of hands to detect.
            detection_conf: Minimum detection confidence threshold.
            tracking_conf: Minimum tracking confidence threshold.
        """
        # Locate the model file
        model_path = self._find_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=tracking_conf,
            min_tracking_confidence=tracking_conf,
        )

        self.detector = HandLandmarker.create_from_options(options)

        self.landmarks = []
        self.hand_detected = False
        self._result = None

    @staticmethod
    def _find_model():
        """Locate the hand_landmarker.task model file."""
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "hand_landmarker.task"),
            os.path.join("assets", "hand_landmarker.task"),
            "hand_landmarker.task",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "hand_landmarker.task model not found. "
            "Please place it in the assets/ directory."
        )

    def find_hands(self, frame, draw=True):
        """
        Process a frame and optionally draw hand landmarks.

        Args:
            frame: BGR image from OpenCV.
            draw: Whether to draw landmarks on the frame.

        Returns:
            The frame (with or without landmarks drawn).
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        self._result = self.detector.detect(mp_image)

        self.hand_detected = bool(
            self._result.hand_landmarks and len(self._result.hand_landmarks) > 0
        )

        if self.hand_detected and draw:
            for hand_lms in self._result.hand_landmarks:
                # Draw connections
                for connection in self.HAND_CONNECTIONS:
                    start_idx = connection.start
                    end_idx = connection.end
                    h, w, _ = frame.shape

                    start_lm = hand_lms[start_idx]
                    end_lm = hand_lms[end_idx]

                    start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pt = (int(end_lm.x * w), int(end_lm.y * h))

                    cv2.line(frame, start_pt, end_pt, (0, 255, 200), 2, cv2.LINE_AA)

                # Draw landmark points
                h, w, _ = frame.shape
                for lm in hand_lms:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 5, (0, 200, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 5, (0, 100, 128), 1, cv2.LINE_AA)

        return frame

    def get_landmarks(self, frame):
        """
        Extract landmark positions as pixel coordinates.

        Args:
            frame: The current frame (used for dimension reference).

        Returns:
            List of (id, x, y) tuples for all 21 landmarks,
            or empty list if no hand detected.
        """
        self.landmarks = []

        if not self.hand_detected or not self._result.hand_landmarks:
            return self.landmarks

        h, w, _ = frame.shape
        hand_lms = self._result.hand_landmarks[0]

        for idx, lm in enumerate(hand_lms):
            px, py = int(lm.x * w), int(lm.y * h)
            self.landmarks.append((idx, px, py))

        return self.landmarks

    def get_finger_states(self):
        """
        Determine which fingers are up (extended).

        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky].
            Returns all False if no hand detected.
        """
        if len(self.landmarks) < 21:
            return [False] * 5

        fingers = []

        # Thumb: compare tip X vs IP joint X
        thumb_tip = self.landmarks[4]
        thumb_ip = self.landmarks[3]
        thumb_mcp = self.landmarks[2]
        wrist = self.landmarks[0]

        # Determine if right or left hand by wrist-to-thumb direction
        if thumb_mcp[1] < wrist[1]:
            fingers.append(thumb_tip[1] < thumb_ip[1])
        else:
            fingers.append(thumb_tip[1] > thumb_ip[1])

        # Index through Pinky: tip Y above PIP Y means finger is up
        # (Y axis is inverted in image coordinates: smaller Y = higher up)
        for tip_id, pip_id in zip(self.FINGERTIP_IDS[1:], self.FINGER_PIP_IDS[1:]):
            tip_y = self.landmarks[tip_id][2]
            pip_y = self.landmarks[pip_id][2]
            fingers.append(tip_y < pip_y)

        return fingers

    def get_fingertip_position(self, finger_index=1):
        """
        Get the position of a specific fingertip.

        Args:
            finger_index: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky

        Returns:
            (x, y) tuple, or None if no hand detected.
        """
        if len(self.landmarks) < 21:
            return None

        tip_id = self.FINGERTIP_IDS[finger_index]
        return (self.landmarks[tip_id][1], self.landmarks[tip_id][2])

    def get_finger_distance(self, finger1=1, finger2=2):
        """
        Calculate distance between two fingertips.

        Args:
            finger1: First finger index.
            finger2: Second finger index.

        Returns:
            Euclidean distance, or -1 if hand not detected.
        """
        pos1 = self.get_fingertip_position(finger1)
        pos2 = self.get_fingertip_position(finger2)

        if pos1 is None or pos2 is None:
            return -1

        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def release(self):
        """Release MediaPipe resources."""
        self.detector.close()
