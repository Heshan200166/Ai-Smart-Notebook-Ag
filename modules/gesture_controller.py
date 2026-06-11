"""
Gesture Controller Module — V3 (Rewritten for Accuracy)
========================================================
Simpler, more forgiving gesture classification with larger stabilization
buffer and weighted voting.

Key changes:
- Larger buffer (7 frames) with weighted recency voting
- Simpler gesture rules with wider tolerance
- Separate "confirmed" gesture that only changes after strong consensus
"""

import time
from collections import deque


class Gesture:
    """Enumeration of recognized gestures."""
    NONE = "none"
    DRAW = "draw"
    SELECT = "select"
    CLEAR = "clear"
    SAVE = "save"
    ERASE = "erase"


class GestureController:
    """
    Translates finger states into application gestures with strong
    stabilization and forgiving classification.
    """

    HOLD_DURATION = 0.8
    COOLDOWN_DURATION = 2.0

    # Larger buffer for stability
    BUFFER_SIZE = 7

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
        self.save_triggered = False

        # --- Stabilization buffer with timestamps ---
        self._gesture_buffer = deque(maxlen=self.BUFFER_SIZE)

        # --- Transition rules ---
        # How many buffer votes needed to switch TO a given gesture
        self._switch_threshold = {
            Gesture.DRAW: 2,      # Quick response for drawing
            Gesture.SELECT: 3,    # Moderate response
            Gesture.ERASE: 3,     # Moderate response
            Gesture.CLEAR: 4,     # Higher threshold (destructive)
            Gesture.SAVE: 4,      # Higher threshold (destructive)
            Gesture.NONE: 3,      # Moderate to stop
        }

    def recognize(self, finger_states):
        """
        Recognize gesture from finger states with weighted stabilization.

        Args:
            finger_states: List of 5 booleans [thumb, index, middle, ring, pinky].

        Returns:
            Current stabilized gesture string.
        """
        if not finger_states or len(finger_states) != 5:
            self._gesture_buffer.append(Gesture.NONE)
            return self._update_gesture()

        # Reset triggered flags
        self.clear_triggered = False
        self.save_triggered = False

        # Classify raw gesture
        raw = self._classify(finger_states)
        self._gesture_buffer.append(raw)

        # Get stabilized gesture
        return self._update_gesture()

    def _classify(self, states):
        """
        Classify gesture from finger states.
        Rules are ordered by specificity (most specific first).

        The thumb is FULLY IGNORED for all gestures because it is
        the most unreliably detected finger across all hand poses.
        """
        _, index, middle, ring, pinky = states

        # Count non-thumb fingers up
        non_thumb_up = sum([index, middle, ring, pinky])

        # === FIST: No non-thumb fingers up → Save ===
        if non_thumb_up == 0:
            return Gesture.SAVE

        # === OPEN PALM: All 4 non-thumb fingers up → Clear ===
        if non_thumb_up == 4:
            return Gesture.CLEAR

        # === INDEX ONLY: Just index up → Draw ===
        if index and non_thumb_up == 1:
            return Gesture.DRAW

        # === PEACE / V-SIGN: Index + Middle up → Select ===
        if index and middle and non_thumb_up == 2:
            return Gesture.SELECT

        # === MIDDLE ONLY: Just middle up → Erase ===
        if middle and non_thumb_up == 1:
            return Gesture.ERASE

        # === 3 fingers up (index+middle+ring) → still treat as Select ===
        if index and middle and ring and not pinky:
            return Gesture.SELECT

        # === Index + any 1 other (not middle) → still Draw ===
        if index and non_thumb_up == 2 and not middle:
            return Gesture.DRAW

        return Gesture.NONE

    def _update_gesture(self):
        """
        Determine the stable gesture using weighted recency voting.
        Recent frames have more weight than older ones.
        """
        if not self._gesture_buffer:
            self.current_gesture = Gesture.NONE
            return self.current_gesture

        now = time.time()
        in_cooldown = (now - self._last_trigger_time) < self.COOLDOWN_DURATION

        # Weighted vote: more recent = higher weight
        # Weights: [1, 1, 2, 2, 3, 3, 4] for buffer of 7
        weights = []
        buf_len = len(self._gesture_buffer)
        for i in range(buf_len):
            weights.append(1 + i * 3 // buf_len)

        # Count weighted votes
        vote_scores = {}
        for weight, gesture in zip(weights, self._gesture_buffer):
            vote_scores[gesture] = vote_scores.get(gesture, 0) + weight

        # Find winner
        winner = max(vote_scores, key=vote_scores.get)
        winner_score = vote_scores[winner]
        total_weight = sum(weights)

        # Calculate vote ratio
        vote_ratio = winner_score / total_weight

        # Determine threshold
        threshold = self._switch_threshold.get(winner, 3)
        raw_count = sum(1 for g in self._gesture_buffer if g == winner)

        # Accept if raw count meets threshold OR weighted ratio is high
        if raw_count >= threshold or vote_ratio > 0.6:
            new_gesture = winner
        else:
            new_gesture = self.current_gesture  # Keep current

        # --- Handle hold gestures ---
        if new_gesture in (Gesture.CLEAR, Gesture.SAVE):
            if not in_cooldown:
                if self._held_gesture == new_gesture:
                    hold_time = now - self._hold_start_time
                    if hold_time >= self.HOLD_DURATION:
                        if new_gesture == Gesture.CLEAR:
                            self.clear_triggered = True
                        elif new_gesture == Gesture.SAVE:
                            self.save_triggered = True
                        self._last_trigger_time = now
                        self._held_gesture = Gesture.NONE
                        self._gesture_buffer.clear()
                else:
                    self._held_gesture = new_gesture
                    self._hold_start_time = now
        else:
            self._held_gesture = Gesture.NONE

        self.previous_gesture = self.current_gesture
        self.current_gesture = new_gesture
        return self.current_gesture

    def get_hold_progress(self):
        """Get the progress of the current hold gesture (0.0 to 1.0)."""
        if self._held_gesture in (Gesture.CLEAR, Gesture.SAVE):
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
            Gesture.SAVE: "✊ Hold to Save",
            Gesture.ERASE: "🧹 Eraser",
        }
        return names.get(self.current_gesture, "Unknown")
