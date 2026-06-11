"""
Gesture Controller Module
==========================
Interprets finger states from HandTracker into application-level gestures.
Includes debounce logic for destructive actions (clear/save) and cooldown timers.
"""

import time


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
    Translates finger states into application gestures with debounce protection.
    """

    # Hold duration required for destructive gestures (seconds)
    HOLD_DURATION = 1.0
    # Cooldown after a destructive gesture triggers (seconds)
    COOLDOWN_DURATION = 2.0

    def __init__(self):
        self.current_gesture = Gesture.NONE
        self.previous_gesture = Gesture.NONE

        # Debounce tracking for hold-to-activate gestures
        self._hold_start_time = 0
        self._held_gesture = Gesture.NONE

        # Cooldown tracking
        self._last_trigger_time = 0

        # Gesture-triggered flags (consumed by the main loop)
        self.clear_triggered = False
        self.save_triggered = False

    def recognize(self, finger_states):
        """
        Recognize gesture from finger states.

        Args:
            finger_states: List of 5 booleans [thumb, index, middle, ring, pinky].

        Returns:
            Current gesture string.
        """
        if not finger_states or len(finger_states) != 5:
            self.current_gesture = Gesture.NONE
            return self.current_gesture

        thumb, index, middle, ring, pinky = finger_states
        fingers_up = sum(finger_states)

        # Reset triggered flags each frame
        self.clear_triggered = False
        self.save_triggered = False

        now = time.time()
        in_cooldown = (now - self._last_trigger_time) < self.COOLDOWN_DURATION

        # --- Gesture Classification ---

        # Open Palm: all 5 fingers up → Clear Canvas (hold to activate)
        if fingers_up == 5:
            raw_gesture = Gesture.CLEAR

        # Fist: all fingers down → Save Drawing (hold to activate)
        elif fingers_up == 0:
            raw_gesture = Gesture.SAVE

        # Index + Middle up, others down → Selection/Move mode
        elif index and middle and not ring and not pinky:
            raw_gesture = Gesture.SELECT

        # Only index finger up → Draw
        elif index and not middle and not ring and not pinky:
            raw_gesture = Gesture.DRAW

        # Thumb + Index pinch (thumb up, others arranged for erase)
        elif thumb and middle and not index and not ring and not pinky:
            raw_gesture = Gesture.ERASE

        else:
            raw_gesture = Gesture.NONE

        # --- Debounce for destructive gestures ---
        if raw_gesture in (Gesture.CLEAR, Gesture.SAVE):
            if not in_cooldown:
                if self._held_gesture == raw_gesture:
                    # Still holding the same gesture
                    hold_time = now - self._hold_start_time
                    if hold_time >= self.HOLD_DURATION:
                        # Trigger!
                        if raw_gesture == Gesture.CLEAR:
                            self.clear_triggered = True
                        elif raw_gesture == Gesture.SAVE:
                            self.save_triggered = True
                        self._last_trigger_time = now
                        self._held_gesture = Gesture.NONE
                else:
                    # New hold gesture started
                    self._held_gesture = raw_gesture
                    self._hold_start_time = now
            self.current_gesture = raw_gesture
        else:
            # Reset hold tracking for non-destructive gestures
            self._held_gesture = Gesture.NONE
            self.current_gesture = raw_gesture

        self.previous_gesture = self.current_gesture
        return self.current_gesture

    def get_hold_progress(self):
        """
        Get the progress of the current hold gesture (0.0 to 1.0).

        Returns:
            Float between 0 and 1 representing hold progress.
        """
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
