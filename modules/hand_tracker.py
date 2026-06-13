"""
Hand Tracker Module — V4 (Orientation-Aware + Body-Part Filtering)
====================================================================
Uses MediaPipe HandLandmarker in VIDEO mode for temporal tracking.

Key improvements:
- Hand validation: rejects false detections on body parts by checking
  landmark structure (spread, proportions, finger lengths)
- Higher detection confidence (0.5) with CLAHE fallback
- Orientation-aware finger detection: works with hand pointing UP or DOWN
  so the user can draw in the full canvas area
- Velocity-adaptive EMA smoothing
"""

import cv2
import numpy as np
import os
import time

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
)


class HandTracker:
    """High-accuracy hand tracking with orientation support."""

    # MediaPipe landmark IDs
    FINGERTIP_IDS = [4, 8, 12, 16, 20]       # Thumb, Index, Middle, Ring, Pinky tips
    FINGER_PIP_IDS = [2, 6, 10, 14, 18]      # PIP/IP joints
    FINGER_MCP_IDS = [1, 5, 9, 13, 17]       # MCP joints

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

    # --- Hand validation thresholds ---
    MIN_HAND_SPAN = 40         # Minimum pixel span (width or height) to accept
    MIN_FINGER_RATIO = 0.15    # Minimum (longest finger / hand span) ratio
    MAX_ASPECT_RATIO = 4.0     # Maximum width/height or height/width ratio

    def __init__(self, max_hands=1, detection_conf=0.5, tracking_conf=0.4,
                 smoothing_factor=0.55, persistence_frames=8):
        """
        Initialize the hand tracker with validation and orientation support.
        """
        model_path = self._find_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

        self.detector = HandLandmarker.create_from_options(options)

        self._timestamp_ms = 0

        self.landmarks = []
        self.hand_detected = False
        self._result = None

        # Frame preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Smoothing
        self._smoothing_factor = smoothing_factor
        self._smoothed_coords = None
        self._prev_coords = None
        self._velocity = None

        # Persistence
        self._persistence_frames = persistence_frames
        self._frames_since_detection = 0
        self._last_valid_coords = None

        # Finger state tracking
        self._finger_states = [False] * 5
        self._finger_confidence = [0.0] * 5

        # Hand orientation (updated each frame)
        self._hand_inverted = False  # True when hand points downward

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

    def _preprocess_frame(self, frame):
        """Enhance frame with CLAHE for better detection."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _validate_hand(self, hand_lms, frame_w, frame_h):
        """
        Validate that detected landmarks form a plausible hand shape.
        Rejects false detections on body parts like arms, shoulders, etc.

        Checks:
        1. Bounding box is large enough (not a tiny noise detection)
        2. Aspect ratio is reasonable for a hand
        3. Finger lengths are proportional (at least one finger is long enough)
        4. Landmarks are spread out (not all clustered in one spot)

        Returns:
            True if the detection looks like a real hand.
        """
        # Convert to pixel coords
        coords = np.array([(lm.x * frame_w, lm.y * frame_h) for lm in hand_lms])

        # --- Check 1: Bounding box size ---
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        span = max_xy - min_xy
        width, height = span[0], span[1]

        if width < self.MIN_HAND_SPAN and height < self.MIN_HAND_SPAN:
            return False  # Too small — likely noise

        # --- Check 2: Aspect ratio ---
        aspect = max(width, height) / (min(width, height) + 1e-6)
        if aspect > self.MAX_ASPECT_RATIO:
            return False  # Too elongated — likely arm/edge

        # --- Check 3: Finger proportions ---
        wrist = coords[0]
        # Check at least one finger (tip-to-MCP) is a reasonable fraction of hand span
        hand_span = max(width, height)
        has_valid_finger = False

        for tip_id, mcp_id in zip(self.FINGERTIP_IDS[1:], self.FINGER_MCP_IDS[1:]):
            finger_len = np.linalg.norm(coords[tip_id] - coords[mcp_id])
            if finger_len / (hand_span + 1e-6) > self.MIN_FINGER_RATIO:
                has_valid_finger = True
                break

        if not has_valid_finger:
            return False  # No recognizable finger structure

        # --- Check 4: Landmark spread ---
        # Calculate standard deviation of landmark positions
        # A real hand has landmarks distributed across the area
        std_x = np.std(coords[:, 0])
        std_y = np.std(coords[:, 1])
        avg_std = (std_x + std_y) / 2

        if avg_std < 10:
            return False  # All landmarks clustered — not a hand

        return True

    def _detect_orientation(self):
        """
        Detect hand orientation by checking the wrist-to-middle-MCP vector.
        If the MCP is below the wrist (in screen coords), hand is inverted.

        Sets self._hand_inverted accordingly.
        """
        if len(self.landmarks) < 21:
            return

        wrist_y = self.landmarks[0][2]       # Landmark 0 = wrist
        middle_mcp_y = self.landmarks[9][2]  # Landmark 9 = middle finger MCP

        # In screen coords, Y increases downward
        # Normal: MCP is above wrist (MCP_y < wrist_y)
        # Inverted: MCP is below wrist (MCP_y > wrist_y)
        self._hand_inverted = middle_mcp_y > wrist_y

    def _detect_hands(self, frame):
        """Run detection. Falls back to CLAHE-enhanced frame on failure."""
        self._timestamp_ms += 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)

        # Validate detection
        h, w, _ = frame.shape
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            if not self._validate_hand(result.hand_landmarks[0], w, h):
                # Failed validation — treat as no detection
                result = type(result)(hand_landmarks=[], hand_world_landmarks=[],
                                      handedness=[])

        # Fallback: try with enhanced frame
        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            enhanced = self._preprocess_frame(frame)
            rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_enhanced)

            self._timestamp_ms += 1
            result = self.detector.detect_for_video(mp_enhanced, self._timestamp_ms)

            # Validate fallback too
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                if not self._validate_hand(result.hand_landmarks[0], w, h):
                    result = type(result)(hand_landmarks=[], hand_world_landmarks=[],
                                          handedness=[])

        return result

    def find_hands(self, frame, draw=True):
        """Process a frame, validate, and optionally draw landmarks."""
        self._result = self._detect_hands(frame)

        raw_detected = bool(
            self._result.hand_landmarks and len(self._result.hand_landmarks) > 0
        )

        if raw_detected:
            self.hand_detected = True
            self._frames_since_detection = 0
        else:
            self._frames_since_detection += 1
            self.hand_detected = self._frames_since_detection <= self._persistence_frames
            if not self.hand_detected:
                self._smoothed_coords = None
                self._prev_coords = None
                self._velocity = None
                self._finger_states = [False] * 5
                self._finger_confidence = [0.0] * 5

        # Draw landmarks
        if raw_detected and draw:
            h, w, _ = frame.shape
            for hand_lms in self._result.hand_landmarks:
                for connection in self.HAND_CONNECTIONS:
                    start_lm = hand_lms[connection.start]
                    end_lm = hand_lms[connection.end]
                    start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(frame, start_pt, end_pt, (0, 255, 200), 2, cv2.LINE_AA)

                for i, lm in enumerate(hand_lms):
                    px, py = int(lm.x * w), int(lm.y * h)
                    if i in self.FINGERTIP_IDS:
                        cv2.circle(frame, (px, py), 8, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)

                # Draw orientation indicator
                if self.landmarks:
                    wrist = self.landmarks[0]
                    indicator_text = "INV" if self._hand_inverted else "NRM"
                    indicator_color = (0, 165, 255) if self._hand_inverted else (0, 255, 0)
                    cv2.putText(frame, indicator_text,
                                (wrist[1] - 15, wrist[2] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, indicator_color, 1, cv2.LINE_AA)

        return frame

    def get_landmarks(self, frame):
        """Extract smoothed landmark positions as pixel coordinates."""
        self.landmarks = []
        h, w, _ = frame.shape

        raw_detected = bool(
            self._result and self._result.hand_landmarks
            and len(self._result.hand_landmarks) > 0
        )

        if raw_detected:
            hand_lms = self._result.hand_landmarks[0]
            raw_coords = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_lms],
                dtype=np.float64
            )

            if self._smoothed_coords is None:
                self._smoothed_coords = raw_coords.copy()
                self._velocity = np.zeros_like(raw_coords)
            else:
                self._velocity = raw_coords - self._smoothed_coords
                speed = np.linalg.norm(self._velocity, axis=1, keepdims=True)
                speed_factor = np.clip(speed / 50.0, 0.0, 1.0)
                alpha = self._smoothing_factor + speed_factor * (0.9 - self._smoothing_factor)
                self._smoothed_coords = (
                    alpha * raw_coords + (1 - alpha) * self._smoothed_coords
                )

            self._last_valid_coords = self._smoothed_coords.copy()
            self._prev_coords = raw_coords.copy()

        elif self.hand_detected and self._last_valid_coords is not None:
            pass
        else:
            return self.landmarks

        coords = self._smoothed_coords if raw_detected else self._last_valid_coords
        if coords is not None:
            for idx in range(21):
                px, py = int(coords[idx][0]), int(coords[idx][1])
                self.landmarks.append((idx, px, py))

        # Detect orientation after landmarks are populated
        self._detect_orientation()

        return self.landmarks

    def get_finger_states(self):
        """
        Determine which fingers are extended using ORIENTATION-AWARE checks.

        Detects whether the hand is upright or inverted (pointing down)
        and adapts the finger detection accordingly. This allows drawing
        in the full canvas area — both upper and lower regions.

        The primary check uses a rotation-invariant approach:
        - "Extended" means tip is FURTHER from the palm center than PIP
        - Combined with directional checks adapted to orientation
        """
        if len(self.landmarks) < 21:
            return [False] * 5

        new_states = []

        # Get palm center (average of wrist + 4 MCP joints)
        palm_pts = [self.landmarks[i][1:3] for i in [0, 5, 9, 13, 17]]
        palm_center = np.mean(palm_pts, axis=0)

        wrist = np.array(self.landmarks[0][1:3])
        index_mcp = np.array(self.landmarks[5][1:3])

        # Direction multiplier: +1 for normal, -1 for inverted
        direction = -1 if self._hand_inverted else 1

        # --- Thumb (special case) ---
        thumb_tip = np.array(self.landmarks[4][1:3])
        thumb_ip = np.array(self.landmarks[3][1:3])
        thumb_mcp = np.array(self.landmarks[2][1:3])

        # Thumb check: tip distance from palm center vs MCP distance
        tip_palm_dist = np.linalg.norm(thumb_tip - palm_center)
        mcp_palm_dist = np.linalg.norm(thumb_mcp - palm_center)
        check1 = tip_palm_dist > mcp_palm_dist * 1.2

        tip_to_index = np.linalg.norm(thumb_tip - index_mcp)
        mcp_to_index = np.linalg.norm(thumb_mcp - index_mcp)
        check2 = tip_to_index > mcp_to_index * 0.9

        v1 = thumb_mcp - wrist
        v2 = thumb_tip - thumb_mcp
        angle = self._angle_between(v1, v2)
        check3 = angle < 160

        thumb_votes = sum([check1, check2, check3])
        thumb_up = thumb_votes >= 2
        thumb_up = self._apply_confidence(0, thumb_up, thumb_votes / 3.0)
        new_states.append(thumb_up)

        # --- Index through Pinky (orientation-aware) ---
        for i, (tip_id, pip_id, mcp_id) in enumerate(
            zip(self.FINGERTIP_IDS[1:], self.FINGER_PIP_IDS[1:], self.FINGER_MCP_IDS[1:]),
            start=1
        ):
            tip = np.array(self.landmarks[tip_id][1:3])
            pip = np.array(self.landmarks[pip_id][1:3])
            mcp = np.array(self.landmarks[mcp_id][1:3])
            wrist_pt = np.array(self.landmarks[0][1:3])

            # === PRIMARY CHECK (orientation-aware) ===
            # Normal: tip Y < PIP Y (tip above PIP)
            # Inverted: tip Y > PIP Y (tip below PIP = extended downward)
            if self._hand_inverted:
                # Inverted: "extended" means tip is BELOW PIP
                if i >= 3:
                    primary_up = tip[1] > (pip[1] + 15)
                else:
                    primary_up = tip[1] > pip[1]
            else:
                # Normal: "extended" means tip is ABOVE PIP
                if i >= 3:
                    primary_up = tip[1] < (pip[1] - 15)
                else:
                    primary_up = tip[1] < pip[1]

            if not primary_up:
                finger_up = False
                vote_conf = 0.0
            else:
                # === SECONDARY CHECKS (rotation-invariant) ===

                # Check 2: Tip further from palm center than PIP
                tip_palm = np.linalg.norm(tip - palm_center)
                pip_palm = np.linalg.norm(pip - palm_center)
                check2 = tip_palm > pip_palm * 0.95

                # Check 3: Tip further from wrist than PIP
                tip_wrist_dist = np.linalg.norm(tip - wrist_pt)
                pip_wrist_dist = np.linalg.norm(pip - wrist_pt)
                if i >= 3:
                    check3 = tip_wrist_dist > pip_wrist_dist * 1.15
                else:
                    check3 = tip_wrist_dist > pip_wrist_dist * 1.0

                confirmation_votes = sum([check2, check3])
                vote_conf = (1.0 + confirmation_votes) / 3.0
                finger_up = True

            finger_up = self._apply_confidence(i, finger_up, vote_conf)
            new_states.append(finger_up)

        self._finger_states = new_states
        return new_states

    def _apply_confidence(self, finger_idx, raw_up, vote_confidence):
        """
        Apply confidence-based hysteresis with per-finger tuning.
        """
        current_up = self._finger_states[finger_idx]
        conf = self._finger_confidence[finger_idx]

        is_index = (finger_idx == 1)
        is_ring_pinky = (finger_idx >= 3)

        if raw_up:
            if is_index:
                rise_rate = 0.5
            elif is_ring_pinky:
                rise_rate = 0.25
            else:
                rise_rate = 0.4
            conf = min(conf + vote_confidence * rise_rate, 1.0)
        else:
            if is_index:
                fall_rate = 0.35
            elif is_ring_pinky:
                fall_rate = 0.5
            else:
                fall_rate = 0.4
            conf = max(conf - max(vote_confidence, 0.3) * fall_rate, 0.0)

        self._finger_confidence[finger_idx] = conf

        if current_up:
            if is_ring_pinky:
                down_threshold = 0.4
            elif is_index:
                down_threshold = 0.25
            else:
                down_threshold = 0.3
            return conf >= down_threshold
        else:
            if is_ring_pinky:
                up_threshold = 0.75
            elif is_index:
                up_threshold = 0.5
            else:
                up_threshold = 0.6
            return conf >= up_threshold

    @staticmethod
    def _angle_between(v1, v2):
        """Calculate angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def get_fingertip_position(self, finger_index=1):
        """Get the smoothed position of a specific fingertip."""
        if len(self.landmarks) < 21:
            return None
        tip_id = self.FINGERTIP_IDS[finger_index]
        return (self.landmarks[tip_id][1], self.landmarks[tip_id][2])

    def get_finger_distance(self, finger1=1, finger2=2):
        """Calculate distance between two fingertips."""
        pos1 = self.get_fingertip_position(finger1)
        pos2 = self.get_fingertip_position(finger2)
        if pos1 is None or pos2 is None:
            return -1
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def release(self):
        """Release MediaPipe resources."""
        self.detector.close()
