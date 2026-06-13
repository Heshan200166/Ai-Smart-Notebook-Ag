"""
Gesture Controller Module — V6 (Pen Up/Down State Machine)
============================================================
Drawing uses a symmetrical hold-to-toggle system:

  1. Show index finger → enters draw mode (pen UP initially)
  2. Hold still 1 second → pen goes DOWN (starts drawing)
  3. Move finger → draws on canvas
  4. Hold still 1 second → pen goes UP (stops drawing)
  5. Move to next letter position freely (no lines drawn)
  6. Hold still 1 second → pen goes DOWN again
  7. Repeat...

Open palm gesture now overrides draw lock immediately.
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
    UNDO = "undo"
    SAVE = "save"


class PenState:
    """Sub-states within the DRAW gesture."""
    UP = "up"           # Pen lifted — movement doesn't draw
    DOWN = "down"       # Pen pressed — movement draws
    TRANSITIONING = "transitioning"  # Holding still, about to switch


class GestureController:
    """
    Translates finger states into application gestures with
    symmetrical pen up/down toggling via stillness.
    """

    HOLD_DURATION = 1.5       # Clear requires 1.5s hold
    COOLDOWN_DURATION = 2.5

    BUFFER_SIZE = 9

    # Frames needed to break out of a gesture via inertia
    DRAW_EXIT_FRAMES = 3
    ERASE_EXIT_FRAMES = 3
    DEFAULT_EXIT_FRAMES = 3

    # --- Stillness config ---
    STILLNESS_TIMEOUT = 1.0       # Seconds to hold still to toggle pen state
    MOVEMENT_THRESHOLD = 8.0      # Pixels — below this = "still"
    RESUME_THRESHOLD = 12.0       # Pixels — above this = "moving" (hysteresis)

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
        self.undo_triggered = False
        self.save_triggered = False

        # --- Stabilization ---
        self._gesture_buffer = deque(maxlen=self.BUFFER_SIZE)

        # --- Inertia tracking ---
        self._different_count = 0
        self._candidate_gesture = Gesture.NONE

        # --- Draw lock ---
        self._draw_active = False
        self._index_down_count = 0

        # --- Pen state machine ---
        self.pen_state = PenState.UP        # Current pen state
        self.draw_paused = False             # True = don't draw this frame
        self._last_draw_pos = None           # Last fingertip (x, y)
        self._still_start_time = None        # When finger became still
        self._is_still = False               # Currently in "still" state
        self._pen_just_toggled = False       # Prevents immediate re-toggle

    def recognize(self, finger_states):
        """
        Recognize gesture from finger states.
        """
        if not finger_states or len(finger_states) != 5:
            self._gesture_buffer.append(Gesture.NONE)
            self.clear_triggered = False
            self.undo_triggered = False
            self.save_triggered = False
            self.draw_paused = True  # No hand = no drawing
            return self._apply_inertia(Gesture.NONE)

        self.clear_triggered = False
        self.undo_triggered = False
        self.save_triggered = False

        # Classify raw gesture
        raw = self._classify(finger_states)
        self._gesture_buffer.append(raw)

        # --- Draw lock logic ---
        _, index, _, _, _ = finger_states

        if self._draw_active:
            # === OPEN PALM OVERRIDE ===
            # If raw classification clearly shows open palm, break draw lock
            # immediately — this is a deliberate gesture
            if raw == Gesture.CLEAR:
                self._draw_active = False
                self._index_down_count = 0
                self._reset_pen_state()
                # Fall through to normal gesture processing below
            elif not index:
                self._index_down_count += 1
                self.draw_paused = True
            else:
                self._index_down_count = 0

            # Exit draw lock if index has been down long enough
            if self._draw_active and self._index_down_count >= self.DRAW_EXIT_FRAMES:
                self._draw_active = False
                self._index_down_count = 0
                self._reset_pen_state()

            # If still in draw lock, enforce DRAW gesture
            if self._draw_active:
                # Set draw_paused based on pen state
                self.draw_paused = (self.pen_state != PenState.DOWN)
                self.current_gesture = Gesture.DRAW
                return Gesture.DRAW

        # Get buffer consensus
        consensus = self._get_consensus()

        # Apply inertia
        result = self._apply_inertia(consensus)

        # Activate draw lock when entering draw mode
        if result == Gesture.DRAW and self.previous_gesture != Gesture.DRAW:
            self._draw_active = True
            self._index_down_count = 0
            self._reset_pen_state()
            self.draw_paused = True  # Start with pen UP — need to hold still first

        # If not drawing, always pause
        if result != Gesture.DRAW:
            self.draw_paused = True

        # Handle CLEAR hold
        self._handle_hold(result)

        self.previous_gesture = self.current_gesture
        self.current_gesture = result
        return result

    def update_draw_position(self, x, y):
        """
        Call every frame during draw mode with the fingertip position.
        Manages the pen up/down state machine based on stillness.
        """
        if not self._draw_active:
            return

        now = time.time()

        if self._last_draw_pos is None:
            self._last_draw_pos = (x, y)
            self._still_start_time = now
            self._is_still = True  # Start as still — user needs to hold to begin
            self._pen_just_toggled = False
            return

        # Calculate movement
        dx = x - self._last_draw_pos[0]
        dy = y - self._last_draw_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        self._last_draw_pos = (x, y)

        # --- Stillness state machine ---
        if self._is_still:
            if distance > self.RESUME_THRESHOLD:
                # Finger started moving
                self._is_still = False
                self._still_start_time = None
                self._pen_just_toggled = False
            else:
                # Still holding still — check timeout
                if self._still_start_time is not None and not self._pen_just_toggled:
                    elapsed = now - self._still_start_time
                    if elapsed >= self.STILLNESS_TIMEOUT:
                        # Toggle pen state!
                        self._toggle_pen()
                        self._pen_just_toggled = True
        else:
            # Currently moving
            if distance < self.MOVEMENT_THRESHOLD:
                # Finger became still
                self._is_still = True
                self._still_start_time = now
                self._pen_just_toggled = False
            # else: still moving, nothing to do

        # Update draw_paused based on pen state
        self.draw_paused = (self.pen_state != PenState.DOWN)

    def _toggle_pen(self):
        """Toggle between pen UP and DOWN."""
        if self.pen_state == PenState.UP:
            self.pen_state = PenState.DOWN  # Start drawing
        else:
            self.pen_state = PenState.UP    # Stop drawing

    def _reset_pen_state(self):
        """Reset all pen/stillness state."""
        self.pen_state = PenState.UP
        self.draw_paused = True
        self._last_draw_pos = None
        self._still_start_time = None
        self._is_still = False
        self._pen_just_toggled = False

    def get_stillness_progress(self):
        """
        Get the progress toward next pen toggle (0.0 to 1.0).
        Returns 0 if not currently still or already toggled.
        """
        if not self._is_still or self._still_start_time is None or self._pen_just_toggled:
            return 0.0
        elapsed = time.time() - self._still_start_time
        return min(elapsed / self.STILLNESS_TIMEOUT, 1.0)

    def _classify(self, states):
        """
        Classify gesture from finger states.
        Thumb is FULLY IGNORED.
        """
        _, index, middle, ring, pinky = states
        non_thumb_up = sum([index, middle, ring, pinky])

        # OPEN PALM: 3 or 4 non-thumb fingers up → Clear
        if non_thumb_up >= 3 and index and middle and ring:
            return Gesture.CLEAR

        # INDEX ONLY: Just index up → Draw
        if index and not middle and not ring and not pinky:
            return Gesture.DRAW

        # PEACE: Index + Middle, no ring/pinky → Select
        if index and middle and not ring and not pinky:
            return Gesture.SELECT

        # MIDDLE ONLY → Erase
        if middle and not index and not ring and not pinky:
            return Gesture.ERASE
            
        # PINKY ONLY → Undo
        if pinky and not index and not middle and not ring:
            return Gesture.UNDO
            
        # ROCK ON (Index + Pinky) → Save
        if index and pinky and not middle and not ring:
            return Gesture.SAVE

        return Gesture.NONE

    def _get_consensus(self):
        """Get consensus gesture from buffer with weighted voting."""
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

        return max(scores, key=scores.get)

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
        """Handle hold-to-activate for gestures requiring confirmation."""
        now = time.time()
        in_cooldown = (now - self._last_trigger_time) < self.COOLDOWN_DURATION

        hold_gestures = [Gesture.CLEAR, Gesture.UNDO, Gesture.SAVE]
        
        # Determine hold duration based on gesture
        # Clear: 1.5s, Save: 2.0s, Undo: 1.0s
        durations = {
            Gesture.CLEAR: 1.5,
            Gesture.SAVE: 2.0,
            Gesture.UNDO: 1.0
        }

        if gesture in hold_gestures and not in_cooldown:
            if self._held_gesture == gesture:
                hold_time = now - self._hold_start_time
                if hold_time >= durations[gesture]:
                    # Trigger the action
                    if gesture == Gesture.CLEAR:
                        self.clear_triggered = True
                    elif gesture == Gesture.SAVE:
                        self.save_triggered = True
                    elif gesture == Gesture.UNDO:
                        self.undo_triggered = True
                        
                    self._last_trigger_time = now
                    self._held_gesture = Gesture.NONE
                    self._gesture_buffer.clear()
            else:
                self._held_gesture = gesture
                self._hold_start_time = now
        else:
            if gesture not in hold_gestures:
                self._held_gesture = Gesture.NONE

    def get_hold_progress(self):
        """Get the progress of the active hold gesture (0.0 to 1.0)."""
        durations = {
            Gesture.CLEAR: 1.5,
            Gesture.SAVE: 2.0,
            Gesture.UNDO: 1.0
        }
        
        if self._held_gesture in durations:
            elapsed = time.time() - self._hold_start_time
            return min(elapsed / durations[self._held_gesture], 1.0)
        return 0.0

    def get_gesture_display_name(self):
        """Get a human-readable name for the current gesture."""
        if self.current_gesture == Gesture.DRAW:
            progress = self.get_stillness_progress()
            if progress > 0.1:
                action = "Pen Down" if self.pen_state == PenState.UP else "Pen Up"
                return f"✏️ Hold → {action} {int(progress * 100)}%"
            elif self.pen_state == PenState.DOWN:
                return "✏️ Drawing (Pen Down)"
            else:
                return "✏️ Pen Up — hold still to draw"

        names = {
            Gesture.NONE: "No Gesture",
            Gesture.SELECT: "👆 Selection (Index+Middle)",
            Gesture.CLEAR: "🖐️ Hold to Clear (Open Palm)",
            Gesture.ERASE: "🧹 Eraser (Middle)",
            Gesture.UNDO: "↩️ Hold to Undo (Pinky)",
            Gesture.SAVE: "💾 Hold to Save (Index+Pinky)",
        }
        return names.get(self.current_gesture, "Unknown")
