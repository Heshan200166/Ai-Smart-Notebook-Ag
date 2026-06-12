"""
Hand Tracker Module — V3 (Rewritten for Accuracy)
===================================================
Uses MediaPipe HandLandmarker in VIDEO mode for temporal tracking,
with frame preprocessing and robust landmark smoothing.

Key changes from V2:
- VIDEO running mode (uses inter-frame tracking for much better accuracy)
- Frame preprocessing: CLAHE contrast enhancement, brightness normalization
- Lower confidence thresholds (0.3) for better range
- Dual-pass detection: retry with enhanced frame on failure
- Weighted EMA smoothing with velocity damping
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
    """High-accuracy hand tracking using MediaPipe VIDEO mode."""

    # MediaPipe landmark IDs
    FINGERTIP_IDS = [4, 8, 12, 16, 20]       # Thumb, Index, Middle, Ring, Pinky tips
    FINGER_PIP_IDS = [2, 6, 10, 14, 18]      # PIP/IP joints
    FINGER_MCP_IDS = [1, 5, 9, 13, 17]       # MCP joints (base of each finger)

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

    def __init__(self, max_hands=1, detection_conf=0.3, tracking_conf=0.3,
                 smoothing_factor=0.55, persistence_frames=8):
        """
        Initialize the hand tracker with VIDEO mode for temporal tracking.

        Args:
            max_hands: Maximum number of hands to detect.
            detection_conf: Minimum detection confidence (low for better catch rate).
            tracking_conf: Minimum tracking confidence.
            smoothing_factor: EMA alpha (0=max smooth, 1=no smooth).
            persistence_frames: Frames to retain landmarks after hand loss.
        """
        model_path = self._find_model()

        # --- VIDEO mode: uses temporal tracking between frames ---
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

        self.detector = HandLandmarker.create_from_options(options)

        # Monotonically increasing timestamp for VIDEO mode
        self._timestamp_ms = 0

        self.landmarks = []
        self.raw_landmarks = []
        self.hand_detected = False
        self._result = None

        # --- Frame preprocessing ---
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # --- Smoothing (Weighted EMA with velocity damping) ---
        self._smoothing_factor = smoothing_factor
        self._smoothed_coords = None  # np.array (21, 2)
        self._prev_coords = None      # For velocity calculation
        self._velocity = None          # np.array (21, 2)

        # --- Persistence ---
        self._persistence_frames = persistence_frames
        self._frames_since_detection = 0
        self._last_valid_coords = None

        # --- Finger state tracking with history ---
        self._finger_states = [False] * 5
        self._finger_confidence = [0.0] * 5  # Running confidence per finger

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
        """
        Enhance frame for better hand detection.
        Applies CLAHE contrast enhancement and brightness normalization.
        """
        # Convert to LAB color space for luminance-only enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to luminance channel
        l_enhanced = self._clahe.apply(l)

        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _detect_hands(self, frame):
        """
        Run detection with VIDEO mode. Falls back to enhanced frame on failure.

        Returns:
            HandLandmarkerResult
        """
        self._timestamp_ms += 33  # ~30 FPS increment

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect_for_video(mp_image, self._timestamp_ms)

        # If no hand found, try with preprocessed frame
        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            enhanced = self._preprocess_frame(frame)
            rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_enhanced)

            self._timestamp_ms += 1  # Increment again (must be monotonic)
            result = self.detector.detect_for_video(mp_enhanced, self._timestamp_ms)

        return result

    def find_hands(self, frame, draw=True):
        """
        Process a frame and optionally draw hand landmarks.

        Args:
            frame: BGR image from OpenCV.
            draw: Whether to draw landmarks on the frame.

        Returns:
            The frame with landmarks drawn.
        """
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
                # Draw connections
                for connection in self.HAND_CONNECTIONS:
                    start_lm = hand_lms[connection.start]
                    end_lm = hand_lms[connection.end]
                    start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                    end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                    cv2.line(frame, start_pt, end_pt, (0, 255, 200), 2, cv2.LINE_AA)

                # Draw landmarks with fingertip highlighting
                for i, lm in enumerate(hand_lms):
                    px, py = int(lm.x * w), int(lm.y * h)
                    if i in self.FINGERTIP_IDS:
                        cv2.circle(frame, (px, py), 8, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (px, py), 4, (0, 200, 255), -1, cv2.LINE_AA)

                # Draw finger state indicators on fingertips
                if self.landmarks:
                    states = self._finger_states
                    for fi, is_up in enumerate(states):
                        tip_id = self.FINGERTIP_IDS[fi]
                        if tip_id < len(self.landmarks):
                            tx, ty = self.landmarks[tip_id][1], self.landmarks[tip_id][2]
                            color = (0, 255, 0) if is_up else (0, 0, 255)
                            cv2.circle(frame, (tx, ty - 15), 5, color, -1, cv2.LINE_AA)

        return frame

    def get_landmarks(self, frame):
        """
        Extract smoothed landmark positions as pixel coordinates.

        Uses velocity-damped EMA: when the hand moves fast, smoothing
        decreases for responsiveness. When still, smoothing increases
        for stability.
        """
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

            # --- Velocity-adaptive EMA ---
            if self._smoothed_coords is None:
                self._smoothed_coords = raw_coords.copy()
                self._velocity = np.zeros_like(raw_coords)
            else:
                # Calculate velocity (movement since last frame)
                self._velocity = raw_coords - self._smoothed_coords

                # Adaptive alpha: faster movement → less smoothing
                speed = np.linalg.norm(self._velocity, axis=1, keepdims=True)
                # Normalize speed: 0-10px = slow, 50+px = fast
                speed_factor = np.clip(speed / 50.0, 0.0, 1.0)

                # Alpha ranges from smoothing_factor (slow) to 0.9 (fast)
                alpha = self._smoothing_factor + speed_factor * (0.9 - self._smoothing_factor)

                self._smoothed_coords = (
                    alpha * raw_coords + (1 - alpha) * self._smoothed_coords
                )

            self._last_valid_coords = self._smoothed_coords.copy()
            self._prev_coords = raw_coords.copy()

        elif self.hand_detected and self._last_valid_coords is not None:
            # Use persisted coords during brief hand loss
            pass
        else:
            return self.landmarks

        # Build landmark list from smoothed or persisted coords
        coords = self._smoothed_coords if raw_detected else self._last_valid_coords
        if coords is not None:
            for idx in range(21):
                px, py = int(coords[idx][0]), int(coords[idx][1])
                self.landmarks.append((idx, px, py))

        return self.landmarks

    def get_finger_states(self):
        """
        Determine which fingers are up using a primary check with
        confirmation checks.

        Logic:
        - Check 1 (tip Y above PIP Y) is REQUIRED for a finger to be "up"
        - Checks 2 and 3 provide confidence weighting
        - Ring and pinky have stricter thresholds (harder to go up)
          to prevent false-positive that triggers CLEAR
        """
        if len(self.landmarks) < 21:
            return [False] * 5

        new_states = []

        # --- Thumb (special case: uses X axis) ---
        thumb_tip = np.array(self.landmarks[4][1:3])
        thumb_ip = np.array(self.landmarks[3][1:3])
        thumb_mcp = np.array(self.landmarks[2][1:3])
        wrist = np.array(self.landmarks[0][1:3])
        index_mcp = np.array(self.landmarks[5][1:3])

        tip_dist = np.linalg.norm(thumb_tip - wrist)
        mcp_dist = np.linalg.norm(thumb_mcp - wrist)
        check1 = tip_dist > mcp_dist * 1.2

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

        # --- Index through Pinky ---
        for i, (tip_id, pip_id, mcp_id) in enumerate(
            zip(self.FINGERTIP_IDS[1:], self.FINGER_PIP_IDS[1:], self.FINGER_MCP_IDS[1:]),
            start=1
        ):
            tip = np.array(self.landmarks[tip_id][1:3])
            pip = np.array(self.landmarks[pip_id][1:3])
            mcp = np.array(self.landmarks[mcp_id][1:3])
            wrist_pt = np.array(self.landmarks[0][1:3])

            # PRIMARY CHECK (required): Tip Y must be above PIP Y
            # For ring (i=3) and pinky (i=4), require a larger margin
            if i >= 3:
                # Ring/Pinky: tip must be clearly above PIP (by at least 15px)
                primary_up = tip[1] < (pip[1] - 15)
            else:
                # Index/Middle: standard check
                primary_up = tip[1] < pip[1]

            if not primary_up:
                # Primary check failed → finger is DOWN, no matter what
                finger_up = False
                vote_conf = 0.0
            else:
                # Primary check passed, use secondary checks for confidence
                # Check 2: Tip Y above MCP Y
                check2 = tip[1] < mcp[1]

                # Check 3: Tip further from wrist than PIP (stricter)
                tip_wrist_dist = np.linalg.norm(tip - wrist_pt)
                pip_wrist_dist = np.linalg.norm(pip - wrist_pt)
                # Ring/Pinky need a much larger margin
                if i >= 3:
                    check3 = tip_wrist_dist > pip_wrist_dist * 1.15
                else:
                    check3 = tip_wrist_dist > pip_wrist_dist * 1.0

                confirmation_votes = sum([check2, check3])
                # With primary + confirmations: total score
                vote_conf = (1.0 + confirmation_votes) / 3.0
                finger_up = True  # Primary passed

            finger_up = self._apply_confidence(i, finger_up, vote_conf)
            new_states.append(finger_up)

        self._finger_states = new_states
        return new_states

    def _apply_confidence(self, finger_idx, raw_up, vote_confidence):
        """
        Apply confidence-based hysteresis to prevent flickering.

        Index finger (idx=1): rises fast, falls at moderate speed
        (no longer ultra-sticky — allows prompt draw stop).

        Ring/Pinky (idx=3,4): rises slowly, falls fast
        (prevents false open-palm detection).
        """
        current_up = self._finger_states[finger_idx]
        conf = self._finger_confidence[finger_idx]

        is_index = (finger_idx == 1)
        is_ring_pinky = (finger_idx >= 3)

        if raw_up:
            if is_index:
                rise_rate = 0.5       # Index rises fast
            elif is_ring_pinky:
                rise_rate = 0.25      # Ring/Pinky rise slowly (prevent false up)
            else:
                rise_rate = 0.4
            conf = min(conf + vote_confidence * rise_rate, 1.0)
        else:
            if is_index:
                fall_rate = 0.35      # Index falls at moderate speed (was 0.15)
            elif is_ring_pinky:
                fall_rate = 0.5       # Ring/Pinky fall fast (eager to go down)
            else:
                fall_rate = 0.4
            conf = max(conf - max(vote_confidence, 0.3) * fall_rate, 0.0)

        self._finger_confidence[finger_idx] = conf

        # Hysteresis thresholds
        if current_up:
            if is_ring_pinky:
                down_threshold = 0.4   # Ring/Pinky drop easily
            elif is_index:
                down_threshold = 0.25  # Index still somewhat sticky
            else:
                down_threshold = 0.3
            return conf >= down_threshold
        else:
            if is_ring_pinky:
                up_threshold = 0.75    # Ring/Pinky very hard to go up
            elif is_index:
                up_threshold = 0.5     # Index goes up readily
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
