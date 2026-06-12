"""
Gesture Controller Module — V4 (Stability-First Rewrite)
==========================================================
Focused on eliminating false triggers and maintaining gesture continuity.

Key changes from V3:
- SAVE gesture REMOVED (too unreliable — use button/Ctrl+S only)
- Draw-mode locking: once drawing starts, stays in draw until index
  finger clearly drops for multiple frames
- Gesture inertia: current gesture gets bonus weight, resists switching
- Simpler classification with wider tolerance
- CLEAR requires very sustained open palm
"""

import time
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
    draw-mode locking and strong inertia to prevent interruptions.
    """

    HOLD_DURATION = 1.5       # Clear requires 1.5s hold (longer = safer)
    COOLDOWN_DURATION = 2.5

    BUFFER_SIZE = 9           # Larger buffer for more stability

    # How many consecutive frames of a DIFFERENT gesture are needed
    # to break out of the current gesture
    DRAW_EXIT_FRAMES = 3      # Reduced from 5 — allows prompt release
    ERASE_EXIT_FRAMES = 3
    DEFAULT_EXIT_FRAMES = 3

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
        # save_triggered REMOVED — save only via button/Ctrl+S

        # --- Stabilization ---
        self._gesture_buffer = deque(maxlen=self.BUFFER_SIZE)

        # --- Inertia tracking ---
        self._different_count = 0
        self._candidate_gesture = Gesture.NONE

        # --- Draw lock ---
        self._draw_active = False   # True when user is actively drawing
        self._index_down_count = 0  # Consecutive frames with index down
        self.draw_paused = False     # True when index is down during draw lock
                                     # Main window should stop drawing but not switch gesture

    def recognize(self, finger_states):
        """
        Recognize gesture from finger states with draw-mode locking
        and inertia-based stabilization.

        Args:
            finger_states: List of 5 booleans [thumb, index, middle, ring, pinky].

        Returns:
            Current stabilized gesture string.
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
                # IMMEDIATELY pause drawing (don't draw with a down finger)
                self.draw_paused = True
            else:
                self._index_down_count = 0

            # Only exit draw lock if index has been down for enough frames
            if self._index_down_count < self.DRAW_EXIT_FRAMES:
                # Stay in draw mode regardless of other classification
                self.current_gesture = Gesture.DRAW
                return Gesture.DRAW
            else:
                # Index finger clearly dropped — release draw lock
                self._draw_active = False
                self._index_down_count = 0

        # Get buffer consensus
        consensus = self._get_consensus()

        # Apply inertia
        result = self._apply_inertia(consensus)

        # Activate draw lock when entering draw mode
        if result == Gesture.DRAW and self.previous_gesture != Gesture.DRAW:
            self._draw_active = True
            self._index_down_count = 0

        # Handle CLEAR hold
        self._handle_hold(result)

        self.previous_gesture = self.current_gesture
        self.current_gesture = result
        return result

    def _classify(self, states):
        """
        Classify gesture from finger states.
        Thumb is FULLY IGNORED (most unreliable finger).
        Rules are simple and non-overlapping.
        """
        _, index, middle, ring, pinky = states
        non_thumb_up = sum([index, middle, ring, pinky])

        # === OPEN PALM: 3 or 4 non-thumb fingers up → Clear ===
        # (3 fingers also counts — sometimes ring/pinky detection is unreliable)
        if non_thumb_up >= 3 and index and middle:
            return Gesture.CLEAR

        # === INDEX ONLY: Just index up (maybe ring/pinky leaking) → Draw ===
        if index and not middle:
            return Gesture.DRAW

        # === PEACE / V-SIGN: Index + Middle up, no ring/pinky → Select ===
        if index and middle and not ring and not pinky:
            return Gesture.SELECT

        # === MIDDLE ONLY: Just middle up → Erase ===
        if middle and not index and not ring and not pinky:
            return Gesture.ERASE

        # === NO FINGERS UP → NONE (NOT save!) ===
        if non_thumb_up == 0:
            return Gesture.NONE

        return Gesture.NONE

    def _get_consensus(self):
        """Get the consensus gesture from the buffer using weighted voting."""
        if not self._gesture_buffer:
            return Gesture.NONE

        buf = list(self._gesture_buffer)
        buf_len = len(buf)

        # Weight recent frames much more heavily
        # e.g. for 9 frames: [1, 1, 1, 2, 2, 2, 3, 3, 4]
        weights = []
        for i in range(buf_len):
            weights.append(1 + (i * 4) // buf_len)

        # Count weighted votes
        scores = {}
        for w, g in zip(weights, buf):
            scores[g] = scores.get(g, 0) + w

        # Give bonus weight to current gesture (inertia)
        if self.current_gesture in scores:
            scores[self.current_gesture] += 2

        winner = max(scores, key=scores.get)
        return winner

    def _apply_inertia(self, new_gesture):
        """
        Apply inertia: resist switching away from current gesture
        unless the new gesture is consistently different for enough frames.
        """
        if new_gesture == self.current_gesture:
            self._different_count = 0
            self._candidate_gesture = Gesture.NONE
            return self.current_gesture

        # New gesture differs from current
        if new_gesture == self._candidate_gesture:
            self._different_count += 1
        else:
            self._candidate_gesture = new_gesture
            self._different_count = 1

        # Determine how many frames needed to switch
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

        # Not enough evidence to switch — keep current
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
        return names.get(self.current_gesture, "Unknown")
