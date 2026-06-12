"""
Gesture Controller Module — V5 (Stillness-Based Draw Release)
===============================================================
Drawing release is now based on finger stillness, not finger-down:
- Drawing continues as long as the index finger keeps moving
- When the finger stops moving (held still) for 1 second → drawing pauses
- If the finger starts moving again → drawing resumes instantly
- Finger-down still works as a fallback release

Other gestures unchanged from V4.
"""

import time
import math
from collections import deque


class Gesture:
    """Enumeration of recognized gestures."""
    NONE = "none"
    DRAW = "draw"
    SELECT = "select"
    CLEAR = "clear"
    ERASE = "erase"


class GestureController:
    """
    Translates finger states into application gestures with
    stillness-based draw release and gesture inertia.
    """

    HOLD_DURATION = 1.5       # Clear requires 1.5s hold
    COOLDOWN_DURATION = 2.5

    BUFFER_SIZE = 9

    # How many consecutive frames of a DIFFERENT gesture to exit
    DRAW_EXIT_FRAMES = 3
    ERASE_EXIT_FRAMES = 3
    DEFAULT_EXIT_FRAMES = 3

    # --- Stillness-based draw release config ---
    STILLNESS_TIMEOUT = 1.0       # Seconds of no movement to pause drawing
    MOVEMENT_THRESHOLD = 8.0      # Pixels — movement below this = "still"
    RESUME_THRESHOLD = 12.0       # Pixels — movement above this = "moving again"
                                  # (higher than MOVEMENT_THRESHOLD for hysteresis)

    def __init__(self):
        self.current_gesture = Gesture.NONE
        self.previous_gesture = Gesture.NONE

        # Debounce for hold gestures
        self._hold_start_time = 0
        self._held_gesture = Gesture.NONE

        # Cooldown
        self._last_trigger_time = 0

        # Triggered flags
        self.clear_triggered = False

        # --- Stabilization ---
        self._gesture_buffer = deque(maxlen=self.BUFFER_SIZE)

        # --- Inertia tracking ---
        self._different_count = 0
        self._candidate_gesture = Gesture.NONE

        # --- Draw lock ---
        self._draw_active = False
        self._index_down_count = 0
        self.draw_paused = False

        # --- Stillness tracking ---
        self._last_draw_pos = None       # (x, y) of fingertip last frame
        self._still_start_time = None    # When the finger first became still
        self._is_still = False           # Currently in "still" state

    def recognize(self, finger_states):
        """
        Recognize gesture from finger states.
        """
        if not finger_states or len(finger_states) != 5:
            self._gesture_buffer.append(Gesture.NONE)
            self.clear_triggered = False
            self.draw_paused = False
            return self._apply_inertia(Gesture.NONE)

        self.clear_triggered = False
        self.draw_paused = False

        # Classify raw gesture
        raw = self._classify(finger_states)
        self._gesture_buffer.append(raw)

        # --- Draw lock logic ---
        _, index, _, _, _ = finger_states
        if self._draw_active:
            if not index:
                self._index_down_count += 1
                self.draw_paused = True
            else:
                self._index_down_count = 0

            # Check stillness-based pause (only when index is up)
            if index and self._is_still and self._still_start_time is not None:
                still_duration = time.time() - self._still_start_time
                if still_duration >= self.STILLNESS_TIMEOUT:
                    self.draw_paused = True

            # Exit draw lock: finger down for enough frames
            if self._index_down_count < self.DRAW_EXIT_FRAMES:
                self.current_gesture = Gesture.DRAW
                return Gesture.DRAW
            else:
                self._draw_active = False
                self._index_down_count = 0
                self._reset_stillness()

        # Get buffer consensus
        consensus = self._get_consensus()

        # Apply inertia
        result = self._apply_inertia(consensus)

        # Activate draw lock when entering draw mode
        if result == Gesture.DRAW and self.previous_gesture != Gesture.DRAW:
            self._draw_active = True
            self._index_down_count = 0
            self._reset_stillness()

        # Handle CLEAR hold
        self._handle_hold(result)

        self.previous_gesture = self.current_gesture
        self.current_gesture = result
        return result

    def update_draw_position(self, x, y):
        """
        Call this every frame when drawing is active with the current
        fingertip position. Updates stillness tracking.

        Args:
            x: Current fingertip X position.
            y: Current fingertip Y position.
        """
        if not self._draw_active:
            return

        now = time.time()

        if self._last_draw_pos is None:
            # First frame of drawing — initialize
            self._last_draw_pos = (x, y)
            self._still_start_time = now
            self._is_still = False
            return

        # Calculate movement distance from last frame
        dx = x - self._last_draw_pos[0]
        dy = y - self._last_draw_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if self._is_still:
            # Currently still — check if movement resumed
            if distance > self.RESUME_THRESHOLD:
                # Finger started moving again → resume drawing
                self._is_still = False
                self._still_start_time = None
                self.draw_paused = False
            # If still below threshold, stay still (timer continues)
        else:
            # Currently moving — check if finger became still
            if distance < self.MOVEMENT_THRESHOLD:
                if self._still_start_time is None:
                    # Just became still — start the timer
                    self._still_start_time = now
                self._is_still = True
            else:
                # Still moving — reset
                self._still_start_time = None

        self._last_draw_pos = (x, y)

    def get_stillness_progress(self):
        """
        Get the progress toward stillness timeout (0.0 to 1.0).
        Used to show a visual indicator to the user.
        """
        if not self._is_still or self._still_start_time is None:
            return 0.0
        elapsed = time.time() - self._still_start_time
        return min(elapsed / self.STILLNESS_TIMEOUT, 1.0)

    def _reset_stillness(self):
        """Reset all stillness tracking state."""
        self._last_draw_pos = None
        self._still_start_time = None
        self._is_still = False

    def _classify(self, states):
        """
        Classify gesture from finger states.
        Thumb is FULLY IGNORED (most unreliable finger).
        """
        _, index, middle, ring, pinky = states
        non_thumb_up = sum([index, middle, ring, pinky])

        # OPEN PALM: 3 or 4 non-thumb fingers up → Clear
        if non_thumb_up >= 3 and index and middle:
            return Gesture.CLEAR

        # INDEX ONLY: Just index up → Draw
        if index and not middle:
            return Gesture.DRAW

        # PEACE / V-SIGN: Index + Middle up, no ring/pinky → Select
        if index and middle and not ring and not pinky:
            return Gesture.SELECT

        # MIDDLE ONLY: Just middle up → Erase
        if middle and not index and not ring and not pinky:
            return Gesture.ERASE

        # No fingers or unrecognized → NONE
        return Gesture.NONE

    def _get_consensus(self):
        """Get the consensus gesture from the buffer using weighted voting."""
        if not self._gesture_buffer:
            return Gesture.NONE

        buf = list(self._gesture_buffer)
        buf_len = len(buf)

        weights = []
        for i in range(buf_len):
            weights.append(1 + (i * 4) // buf_len)

        scores = {}
        for w, g in zip(weights, buf):
            scores[g] = scores.get(g, 0) + w

        if self.current_gesture in scores:
            scores[self.current_gesture] += 2

        winner = max(scores, key=scores.get)
        return winner

    def _apply_inertia(self, new_gesture):
        """Resist switching away from current gesture."""
        if new_gesture == self.current_gesture:
            self._different_count = 0
            self._candidate_gesture = Gesture.NONE
            return self.current_gesture

        if new_gesture == self._candidate_gesture:
            self._different_count += 1
        else:
            self._candidate_gesture = new_gesture
            self._different_count = 1

        if self.current_gesture == Gesture.DRAW:
            exit_frames = self.DRAW_EXIT_FRAMES
        elif self.current_gesture == Gesture.ERASE:
            exit_frames = self.ERASE_EXIT_FRAMES
        else:
            exit_frames = self.DEFAULT_EXIT_FRAMES

        if self._different_count >= exit_frames:
            self._different_count = 0
            self._candidate_gesture = Gesture.NONE
            return new_gesture

        return self.current_gesture

    def _handle_hold(self, gesture):
        """Handle hold-to-activate for CLEAR gesture."""
        now = time.time()
        in_cooldown = (now - self._last_trigger_time) < self.COOLDOWN_DURATION

        if gesture == Gesture.CLEAR and not in_cooldown:
            if self._held_gesture == Gesture.CLEAR:
                hold_time = now - self._hold_start_time
                if hold_time >= self.HOLD_DURATION:
                    self.clear_triggered = True
                    self._last_trigger_time = now
                    self._held_gesture = Gesture.NONE
                    self._gesture_buffer.clear()
            else:
                self._held_gesture = Gesture.CLEAR
                self._hold_start_time = now
        else:
            if gesture != Gesture.CLEAR:
                self._held_gesture = Gesture.NONE

    def get_hold_progress(self):
        """Get the progress of the current hold gesture (0.0 to 1.0)."""
        if self._held_gesture == Gesture.CLEAR:
            elapsed = time.time() - self._hold_start_time
            return min(elapsed / self.HOLD_DURATION, 1.0)
        return 0.0

    def get_gesture_display_name(self):
        """Get a human-readable name for the current gesture."""
        names = {
            Gesture.NONE: "No Gesture",
            Gesture.DRAW: "✏️ Drawing",
            Gesture.SELECT: "👆 Selection",
            Gesture.CLEAR: "🖐️ Hold to Clear",
            Gesture.ERASE: "🧹 Eraser",
        }
        gesture_name = names.get(self.current_gesture, "Unknown")

        # Show stillness status when drawing
        if self.current_gesture == Gesture.DRAW:
            progress = self.get_stillness_progress()
            if progress > 0.1:
                gesture_name = f"✏️ Pausing... {int(progress * 100)}%"
            elif self.draw_paused:
                gesture_name = "✏️ Paused"

        return gesture_name
