"""
Main Window
=============
The primary PyQt6 application window for AI Smart Air Notebook.
Integrates camera feed, hand tracking, gesture recognition, drawing engine,
and a polished dark-themed UI with tool controls.
"""

import cv2
import sys
import os
import time
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFrame, QStatusBar,
    QMessageBox, QGroupBox, QGridLayout, QProgressBar,
    QSizePolicy, QApplication, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon, QFont, QAction

from ui.canvas_widget import CanvasWidget
from modules.hand_tracker import HandTracker
from modules.gesture_controller import GestureController, Gesture
from modules.drawing_engine import DrawingEngine
from modules.database import NotebookDatabase
from modules.ai_worker import AIWorker
from modules.shape_recognizer import ShapeRecognizer


class MainWindow(QMainWindow):
    """Main application window for AI Smart Air Notebook."""

    # Header bar constants for OpenCV HUD
    HEADER_HEIGHT = 80
    COLOR_CIRCLE_RADIUS = 18
    COLOR_CIRCLE_Y = 40
    COLOR_CIRCLE_START_X = 40
    COLOR_CIRCLE_SPACING = 55

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Smart Air Notebook")
        self.setMinimumSize(1200, 800)

        # --- Initialize modules ---
        self.hand_tracker = HandTracker(max_hands=1)
        self.gesture_controller = GestureController()
        self.drawing_engine = DrawingEngine()
        self.database = NotebookDatabase()
        self.shape_recognizer = ShapeRecognizer()

        # --- AI Worker (background thread) ---
        self.ai_worker = AIWorker(self)
        self.ai_worker.text_recognized.connect(self._on_text_recognized)
        self.ai_worker.math_solved.connect(self._on_math_solved)
        self.ai_worker.shape_detected.connect(self._on_shape_detected)
        self.ai_worker.status_message.connect(
            lambda msg: self.status_bar.showMessage(msg, 5000)
        )
        self.ai_worker.error_occurred.connect(
            lambda msg: self.status_bar.showMessage(msg, 5000)
        )

        # --- Selection state ---
        self._selection_start = None
        self._selection_end = None
        self._last_ocr_text = ""

        # --- Session state ---
        self.session_id = self.database.create_session()
        self.camera_active = False
        self.cap = None

        # --- FPS tracking ---
        self._frame_times = []
        self._fps = 0

        # --- Build UI ---
        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()
        self._apply_stylesheet()

        # --- Camera timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)

        # Auto-start camera
        self._start_camera()

    # =========================================================================
    #  UI Setup
    # =========================================================================

    def _setup_ui(self):
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # --- Left: Camera/Canvas display ---
        canvas_container = QFrame()
        canvas_container.setObjectName("canvasContainer")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_widget = CanvasWidget()
        canvas_layout.addWidget(self.canvas_widget)

        main_layout.addWidget(canvas_container, stretch=4)

        # --- Right: Tool panel ---
        tool_panel = QFrame()
        tool_panel.setObjectName("toolPanel")
        tool_panel.setFixedWidth(260)
        tool_layout = QVBoxLayout(tool_panel)
        tool_layout.setContentsMargins(16, 16, 16, 16)
        tool_layout.setSpacing(12)

        # App title in panel
        title_label = QLabel("✨ Smart Notebook")
        title_label.setObjectName("panelTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tool_layout.addWidget(title_label)

        # Separator
        tool_layout.addWidget(self._create_separator())

        # --- Gesture indicator ---
        gesture_group = QGroupBox("Current Gesture")
        gesture_group.setObjectName("gestureGroup")
        gesture_layout = QVBoxLayout(gesture_group)

        self.gesture_label = QLabel("No Gesture")
        self.gesture_label.setObjectName("gestureLabel")
        self.gesture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gesture_layout.addWidget(self.gesture_label)

        self.hold_progress = QProgressBar()
        self.hold_progress.setObjectName("holdProgress")
        self.hold_progress.setRange(0, 100)
        self.hold_progress.setValue(0)
        self.hold_progress.setTextVisible(False)
        self.hold_progress.setFixedHeight(8)
        gesture_layout.addWidget(self.hold_progress)

        tool_layout.addWidget(gesture_group)

        # --- Color palette ---
        color_group = QGroupBox("Color Palette")
        color_group.setObjectName("colorGroup")
        color_grid = QGridLayout(color_group)
        color_grid.setSpacing(8)

        self.color_buttons = []
        for i, (name, bgr) in enumerate(DrawingEngine.COLORS.items()):
            btn = QPushButton()
            btn.setFixedSize(45, 45)
            btn.setToolTip(name)
            r, g, b = bgr[2], bgr[1], bgr[0]  # BGR → RGB
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({r},{g},{b});
                    border: 3px solid transparent;
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    border: 3px solid #ffffff;
                }}
            """)
            btn.clicked.connect(lambda checked, idx=i: self._on_color_selected(idx))
            color_grid.addWidget(btn, i // 4, i % 4)
            self.color_buttons.append(btn)

        tool_layout.addWidget(color_group)

        # Highlight default color
        self._highlight_color_button(0)

        # --- Brush size ---
        size_group = QGroupBox("Brush Size")
        size_group.setObjectName("sizeGroup")
        size_layout = QVBoxLayout(size_group)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setMinimum(0)
        self.size_slider.setMaximum(len(DrawingEngine.BRUSH_SIZES) - 1)
        self.size_slider.setValue(1)
        self.size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.size_slider.setTickInterval(1)
        self.size_slider.valueChanged.connect(self._on_brush_size_changed)
        size_layout.addWidget(self.size_slider)

        self.size_label = QLabel(f"Size: {DrawingEngine.BRUSH_SIZES[1]}px")
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        size_layout.addWidget(self.size_label)

        tool_layout.addWidget(size_group)

        # --- Tool buttons ---
        tools_group = QGroupBox("Tools")
        tools_group.setObjectName("toolsGroup")
        tools_layout = QVBoxLayout(tools_group)
        tools_layout.setSpacing(8)

        self.eraser_btn = QPushButton("🧹  Eraser (🖕 Middle)")
        self.eraser_btn.setObjectName("eraserBtn")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setFixedHeight(40)
        self.eraser_btn.clicked.connect(self._on_eraser_toggled)
        tools_layout.addWidget(self.eraser_btn)

        self.undo_btn = QPushButton("↩️  Undo (🤘 Pinky)")
        self.undo_btn.setObjectName("undoBtn")
        self.undo_btn.setFixedHeight(40)
        self.undo_btn.clicked.connect(self._on_undo)
        tools_layout.addWidget(self.undo_btn)

        self.clear_btn = QPushButton("🗑️  Clear Canvas (🖐️ Open Palm)")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.clicked.connect(self._on_clear)
        tools_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("💾  Save Drawing (🤘 Index+Pinky)")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.setFixedHeight(40)
        self.save_btn.clicked.connect(self._on_save)
        tools_layout.addWidget(self.save_btn)

        tool_layout.addWidget(tools_group)

        # --- AI Features ---
        ai_group = QGroupBox("🧠 AI Features")
        ai_group.setObjectName("aiGroup")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(8)

        self.auto_snap_btn = QPushButton("📐  Shape Auto-Snap: OFF")
        self.auto_snap_btn.setObjectName("autoSnapBtn")
        self.auto_snap_btn.setCheckable(True)
        self.auto_snap_btn.setFixedHeight(36)
        self.auto_snap_btn.clicked.connect(self._on_auto_snap_toggled)
        ai_layout.addWidget(self.auto_snap_btn)

        self.read_text_btn = QPushButton("📝  Read Text (Select Area)")
        self.read_text_btn.setObjectName("readTextBtn")
        self.read_text_btn.setFixedHeight(36)
        self.read_text_btn.clicked.connect(self._on_read_text)
        ai_layout.addWidget(self.read_text_btn)

        self.solve_math_btn = QPushButton("🧮  Solve Math")
        self.solve_math_btn.setObjectName("solveMathBtn")
        self.solve_math_btn.setFixedHeight(36)
        self.solve_math_btn.clicked.connect(self._on_solve_math)
        ai_layout.addWidget(self.solve_math_btn)

        # AI results display
        self.ai_results = QTextEdit()
        self.ai_results.setObjectName("aiResults")
        self.ai_results.setReadOnly(True)
        self.ai_results.setFixedHeight(120)
        self.ai_results.setPlaceholderText("AI results will appear here...")
        ai_layout.addWidget(self.ai_results)

        tool_layout.addWidget(ai_group)

        # --- Camera controls ---
        cam_group = QGroupBox("Camera")
        cam_group.setObjectName("camGroup")
        cam_layout = QVBoxLayout(cam_group)

        self.cam_btn = QPushButton("📷  Stop Camera")
        self.cam_btn.setObjectName("camBtn")
        self.cam_btn.setFixedHeight(40)
        self.cam_btn.clicked.connect(self._toggle_camera)
        cam_layout.addWidget(self.cam_btn)

        tool_layout.addWidget(cam_group)

        # Spacer
        tool_layout.addStretch()

        main_layout.addWidget(tool_panel)

    def _setup_menubar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_session = QAction("New Session", self)
        new_session.setShortcut("Ctrl+N")
        new_session.triggered.connect(self._new_session)
        file_menu.addAction(new_session)

        save_action = QAction("Save Drawing", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        clear_action = QAction("Clear Canvas", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self._on_clear)
        tools_menu.addAction(clear_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._on_undo)
        tools_menu.addAction(undo_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        gestures_action = QAction("Gesture Guide", self)
        gestures_action.triggered.connect(self._show_gesture_guide)
        help_menu.addAction(gestures_action)

    def _setup_statusbar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("fpsLabel")
        self.tool_label = QLabel("Tool: Draw")
        self.tool_label.setObjectName("toolLabel")
        self.session_label = QLabel(f"Session: #{self.session_id}")
        self.session_label.setObjectName("sessionLabel")

        self.status_bar.addWidget(self.fps_label)
        self.status_bar.addWidget(self._create_vseparator())
        self.status_bar.addWidget(self.tool_label)
        self.status_bar.addWidget(self._create_vseparator())
        self.status_bar.addPermanentWidget(self.session_label)

    # =========================================================================
    #  Camera & Processing Loop
    # =========================================================================

    def _start_camera(self):
        """Start the webcam capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(
                self, "Camera Error",
                "Could not open webcam. Please check your camera connection."
            )
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_active = True
        self.cam_btn.setText("📷  Stop Camera")
        self.timer.start(33)  # ~30 FPS

    def _stop_camera(self):
        """Stop the webcam capture."""
        self.timer.stop()
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.setText("📷  Start Camera")

    def _toggle_camera(self):
        """Toggle camera on/off."""
        if self.camera_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _process_frame(self):
        """Main processing loop: capture → track → gesture → draw → display."""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        frame_start = time.time()

        # --- Step 1: Hand tracking ---
        frame = self.hand_tracker.find_hands(frame, draw=True)
        landmarks = self.hand_tracker.get_landmarks(frame)

        # --- Step 2: Gesture recognition ---
        finger_states = self.hand_tracker.get_finger_states()
        gesture = self.gesture_controller.recognize(finger_states)

        # --- Step 3: Process gesture actions ---
        if gesture == Gesture.DRAW:
            pos = self.hand_tracker.get_fingertip_position(1)  # Index finger

            if pos:
                # Update stillness tracking with current fingertip position
                self.gesture_controller.update_draw_position(pos[0], pos[1])

            # Check if draw is paused (stillness timeout OR finger dropped)
            if self.gesture_controller.draw_paused:
                # Stroke just ended — check for shape auto-snap
                if self.drawing_engine.auto_snap_enabled:
                    stroke_pts = self.drawing_engine.get_last_stroke()
                    if len(stroke_pts) >= 8:
                        result = self.shape_recognizer.analyze_stroke(stroke_pts)
                        if result.shape_type != "unknown":
                            # Undo the rough stroke and replace with perfect shape
                            self.drawing_engine.undo()
                            self.drawing_engine._save_undo_state()
                            self.shape_recognizer.render_shape(
                                self.drawing_engine.canvas, result,
                                self.drawing_engine.color,
                                self.drawing_engine.brush_size
                            )
                            self.status_bar.showMessage(
                                f"📐 Shape snapped: {result.shape_type} ({result.confidence:.0%})", 3000
                            )
                self.drawing_engine.stop_drawing()
            else:
                if pos:
                    if pos[1] > self.HEADER_HEIGHT:
                        self.drawing_engine.set_eraser(False)
                        self.drawing_engine.draw(pos[0], pos[1])
                    else:
                        self.drawing_engine.stop_drawing()
                else:
                    self.drawing_engine.stop_drawing()

            # Draw stillness progress ring around fingertip
            if pos:
                stillness = self.gesture_controller.get_stillness_progress()
                if stillness > 0.05:
                    self._draw_stillness_ring(frame, pos, stillness)

        elif gesture == Gesture.ERASE:
            pos = self.hand_tracker.get_fingertip_position(2)  # Middle finger
            if pos and pos[1] > self.HEADER_HEIGHT:
                self.drawing_engine.set_eraser(True)
                self.drawing_engine.draw(pos[0], pos[1])
            else:
                self.drawing_engine.stop_drawing()

        elif gesture == Gesture.SELECT:
            pos = self.hand_tracker.get_fingertip_position(1)  # Index finger
            if pos:
                if pos[1] <= self.HEADER_HEIGHT:
                    # Header area — color/size selection
                    self._check_header_selection(pos[0], pos[1])
                    self._selection_start = None
                    self._selection_end = None
                else:
                    # Canvas area — drag selection rectangle
                    if self._selection_start is None:
                        self._selection_start = (int(pos[0]), int(pos[1]))
                    self._selection_end = (int(pos[0]), int(pos[1]))
            self.drawing_engine.stop_drawing()

        else:
            # If we were selecting and just stopped, finalize the selection
            if self._selection_start is not None and self._selection_end is not None:
                sx, sy = self._selection_start
                ex, ey = self._selection_end
                w_sel = abs(ex - sx)
                h_sel = abs(ey - sy)
                if w_sel > 30 and h_sel > 30:
                    self.drawing_engine.selection_start = self._selection_start
                    self.drawing_engine.selection_end = self._selection_end
                    self.drawing_engine.selection_active = True
                    self.status_bar.showMessage(
                        f"✅ Area selected ({w_sel}x{h_sel}px) — Use 'Read Text' or 'Solve Math'", 5000
                    )
            self._selection_start = None
            self._selection_end = None
            self.drawing_engine.stop_drawing()

        # Handle debounced destructive gestures
        if self.gesture_controller.clear_triggered:
            self.drawing_engine.clear_canvas()
            self.status_bar.showMessage("Canvas cleared via gesture.", 3000)
            
        if self.gesture_controller.undo_triggered:
            self.drawing_engine.undo()
            self.status_bar.showMessage("Undo triggered via gesture.", 3000)
            
        if self.gesture_controller.save_triggered:
            self._on_save()
            self.status_bar.showMessage("Canvas saved via gesture.", 3000)

        # --- Step 4: Draw HUD on frame ---
        frame = self._draw_hud(frame, gesture)

        # --- Step 4b: Draw selection rectangle ---
        if self._selection_start is not None and self._selection_end is not None:
            frame = self.drawing_engine.draw_selection_rect(
                frame,
                self._selection_start[0], self._selection_start[1],
                self._selection_end[0], self._selection_end[1]
            )
        elif self.drawing_engine.selection_active:
            # Show persisted selection
            frame = self.drawing_engine.draw_selection_rect(
                frame,
                self.drawing_engine.selection_start[0],
                self.drawing_engine.selection_start[1],
                self.drawing_engine.selection_end[0],
                self.drawing_engine.selection_end[1]
            )

        # --- Step 5: Overlay canvas onto frame ---
        frame = self.drawing_engine.get_overlay(frame)

        # --- Step 6: Display ---
        self.canvas_widget.update_frame(frame)

        # --- Update UI ---
        self._update_fps(frame_start)
        self.gesture_label.setText(self.gesture_controller.get_gesture_display_name())

        # Update hold progress bar
        progress = self.gesture_controller.get_hold_progress()
        self.hold_progress.setValue(int(progress * 100))

        # Update tool label
        if self.drawing_engine.eraser_mode:
            self.tool_label.setText("Tool: Eraser")
        elif gesture == Gesture.SELECT:
            self.tool_label.setText("Tool: Select")
        else:
            self.tool_label.setText("Tool: Draw")

    # =========================================================================
    #  HUD Drawing (OpenCV overlay on camera frame)
    # =========================================================================

    def _draw_hud(self, frame, gesture):
        """Draw the on-screen header bar with color palette, tools, and debug info."""
        h, w = frame.shape[:2]

        # Semi-transparent header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.HEADER_HEIGHT), (20, 20, 30), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Color circles
        for i, color in enumerate(DrawingEngine.COLOR_LIST):
            cx = self.COLOR_CIRCLE_START_X + i * self.COLOR_CIRCLE_SPACING
            cy = self.COLOR_CIRCLE_Y

            # Draw outer ring for selected color
            if i == self.drawing_engine.color_index:
                cv2.circle(frame, (cx, cy), self.COLOR_CIRCLE_RADIUS + 4,
                           (255, 255, 255), 2, cv2.LINE_AA)

            cv2.circle(frame, (cx, cy), self.COLOR_CIRCLE_RADIUS,
                       color, -1, cv2.LINE_AA)

        # Brush size selector
        size_start_x = self.COLOR_CIRCLE_START_X + len(DrawingEngine.COLOR_LIST) * self.COLOR_CIRCLE_SPACING + 30
        
        for i, size in enumerate(DrawingEngine.BRUSH_SIZES):
            cx = size_start_x + i * 40
            cy = self.COLOR_CIRCLE_Y
            
            # Draw outer ring for selected size
            if i == self.drawing_engine.brush_size_index:
                cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 2, cv2.LINE_AA)
                
            # Draw actual brush size
            cv2.circle(frame, (cx, cy), size, self.drawing_engine.color, -1, cv2.LINE_AA)

        cv2.putText(frame, "Size", (size_start_x - 15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Eraser indicator
        eraser_x = size_start_x + len(DrawingEngine.BRUSH_SIZES) * 40 + 30
        eraser_color = (0, 200, 255) if self.drawing_engine.eraser_mode else (100, 100, 100)
        cv2.rectangle(frame, (eraser_x - 25, 20), (eraser_x + 25, 55), eraser_color, -1)
        cv2.putText(frame, "ERA", (eraser_x - 18, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Gesture hint text (right side)
        hint = self.gesture_controller.get_gesture_display_name()
        text_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, hint, (w - text_size[0] - 20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

        # --- Finger state debug overlay (bottom-left) ---
        finger_states = self.hand_tracker.get_finger_states()
        finger_labels = ["THM", "IDX", "MID", "RNG", "PNK"]
        debug_y = h - 40
        debug_x = 20

        # Background (wider to fit orientation label)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (10, h - 60), (360, h - 10), (20, 20, 30), -1)
        frame = cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0)

        for i, (label, is_up) in enumerate(zip(finger_labels, finger_states)):
            x = debug_x + i * 52
            color = (0, 255, 0) if is_up else (0, 0, 180)
            cv2.circle(frame, (x + 10, debug_y), 8, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x - 2, debug_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 200), 1)

        # Orientation indicator
        orient_x = debug_x + 5 * 52 + 10
        if self.hand_tracker._hand_inverted:
            cv2.putText(frame, "INV", (orient_x, debug_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.putText(frame, "v", (orient_x + 10, debug_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            cv2.putText(frame, "NRM", (orient_x, debug_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "^", (orient_x + 10, debug_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return frame

    def _draw_stillness_ring(self, frame, pos, progress):
        """
        Draw a visual ring around the fingertip showing stillness progress.
        Color depends on what the hold will DO:
        - Green ring = about to START drawing (pen is currently UP)
        - Red ring = about to STOP drawing (pen is currently DOWN)

        Args:
            frame: The current frame to draw on.
            pos: (x, y) fingertip position.
            progress: Float 0.0–1.0 representing stillness progress.
        """
        from modules.gesture_controller import PenState

        x, y = int(pos[0]), int(pos[1])
        radius = 28

        # Color based on pen state (what will happen when hold completes)
        pen_is_down = self.gesture_controller.pen_state == PenState.DOWN

        if pen_is_down:
            # Currently drawing → hold will STOP → red theme
            color = (0, 0, 255)       # Red (BGR)
            label = "Stop"
        else:
            # Currently not drawing → hold will START → green theme
            color = (0, 220, 0)       # Green (BGR)
            label = "Start"

        # Fade color intensity with progress
        intensity = 0.4 + 0.6 * progress
        draw_color = tuple(int(c * intensity) for c in color)

        # Draw background ring (dark)
        cv2.circle(frame, (x, y), radius, (40, 40, 40), 2, cv2.LINE_AA)

        # Draw progress arc
        angle = int(progress * 360)
        if angle > 0:
            cv2.ellipse(
                frame, (x, y), (radius, radius),
                -90, 0, angle,
                draw_color, 3, cv2.LINE_AA
            )

        # Draw action label and percentage
        pct_text = f"{label} {int(progress * 100)}%"
        text_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(
            frame, pct_text,
            (x - text_size[0] // 2, y + radius + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1, cv2.LINE_AA
        )

    def _check_header_selection(self, x, y):
        """Check if the selection point hits a header UI element."""
        # Check color circles
        for i in range(len(DrawingEngine.COLOR_LIST)):
            cx = self.COLOR_CIRCLE_START_X + i * self.COLOR_CIRCLE_SPACING
            cy = self.COLOR_CIRCLE_Y
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist <= self.COLOR_CIRCLE_RADIUS + 5:
                self._on_color_selected(i)
                return
                
        # Check brush size circles
        size_start_x = self.COLOR_CIRCLE_START_X + len(DrawingEngine.COLOR_LIST) * self.COLOR_CIRCLE_SPACING + 30
        for i in range(len(DrawingEngine.BRUSH_SIZES)):
            cx = size_start_x + i * 40
            cy = self.COLOR_CIRCLE_Y
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist <= 18 + 5:
                # Triggering the slider will update the drawing engine
                self.size_slider.setValue(i)
                return

    # =========================================================================
    #  Tool Callbacks
    # =========================================================================

    def _on_color_selected(self, index):
        """Handle color selection."""
        self.drawing_engine.set_color(index)
        self.drawing_engine.eraser_mode = False
        self.eraser_btn.setChecked(False)
        self._highlight_color_button(index)

    def _highlight_color_button(self, selected_index):
        """Update color button borders to show selection."""
        for i, btn in enumerate(self.color_buttons):
            bgr = DrawingEngine.COLOR_LIST[i]
            r, g, b = bgr[2], bgr[1], bgr[0]
            if i == selected_index:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({r},{g},{b});
                        border: 3px solid #ffffff;
                        border-radius: 8px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: rgb({r},{g},{b});
                        border: 3px solid transparent;
                        border-radius: 8px;
                    }}
                    QPushButton:hover {{
                        border: 3px solid #ffffff;
                    }}
                """)

    def _on_brush_size_changed(self, value):
        """Handle brush size slider change."""
        self.drawing_engine.set_brush_size(value)
        self.size_label.setText(f"Size: {DrawingEngine.BRUSH_SIZES[value]}px")

    def _on_eraser_toggled(self):
        """Handle eraser button toggle."""
        self.drawing_engine.toggle_eraser()
        self.eraser_btn.setChecked(self.drawing_engine.eraser_mode)

    def _on_undo(self):
        """Handle undo action."""
        self.drawing_engine.undo()

    def _on_clear(self):
        """Handle clear canvas action."""
        self.drawing_engine.clear_canvas()

    def _on_save(self):
        """Handle save drawing action."""
        if not self.drawing_engine.has_content():
            self.status_bar.showMessage("Nothing to save — canvas is empty.", 3000)
            return

        filepath = self.drawing_engine.save_canvas("sessions")
        self.database.save_drawing(self.session_id, filepath)
        self.status_bar.showMessage(f"Drawing saved: {filepath}", 5000)

    def _new_session(self):
        """Create a new notebook session."""
        self.session_id = self.database.create_session()
        self.drawing_engine.clear_canvas()
        self.session_label.setText(f"Session: #{self.session_id}")
        self.status_bar.showMessage("New session created.", 3000)

    # =========================================================================
    #  AI Feature Callbacks
    # =========================================================================

    def _on_auto_snap_toggled(self):
        """Toggle shape auto-snap mode."""
        enabled = self.auto_snap_btn.isChecked()
        self.drawing_engine.auto_snap_enabled = enabled
        if enabled:
            self.auto_snap_btn.setText("📐  Shape Auto-Snap: ON")
            self.status_bar.showMessage("Shape auto-snap enabled — draw shapes and they'll snap!", 3000)
        else:
            self.auto_snap_btn.setText("📐  Shape Auto-Snap: OFF")
            self.status_bar.showMessage("Shape auto-snap disabled.", 3000)

    def _on_read_text(self):
        """Trigger OCR on the selected canvas region."""
        if self.ai_worker.is_busy():
            self.status_bar.showMessage("⏳ AI is already processing...", 3000)
            return

        if self.drawing_engine.selection_active:
            # Use the selection rectangle
            sx, sy = self.drawing_engine.selection_start
            ex, ey = self.drawing_engine.selection_end
            region = self.drawing_engine.get_region(sx, sy, ex, ey)
        else:
            # Use the entire canvas
            if not self.drawing_engine.has_content():
                self.status_bar.showMessage("Canvas is empty — nothing to read.", 3000)
                return
            region = self.drawing_engine.canvas.copy()

        if region is None:
            self.status_bar.showMessage("⚠️ Selection area too small.", 3000)
            return

        self.ai_worker.process_ocr(region)

    def _on_solve_math(self):
        """Trigger math solving on OCR text or selected region."""
        if self.ai_worker.is_busy():
            self.status_bar.showMessage("⏳ AI is already processing...", 3000)
            return

        # If we have OCR text already, solve it directly
        if self._last_ocr_text and not self._last_ocr_text.startswith("["):
            self.ai_worker.process_math(self._last_ocr_text)
        elif self.drawing_engine.selection_active:
            # OCR + Math combined
            sx, sy = self.drawing_engine.selection_start
            ex, ey = self.drawing_engine.selection_end
            region = self.drawing_engine.get_region(sx, sy, ex, ey)
            if region is not None:
                self.ai_worker.process_ocr_and_math(region)
            else:
                self.status_bar.showMessage("⚠️ Selection area too small.", 3000)
        elif self.drawing_engine.has_content():
            # Use entire canvas
            self.ai_worker.process_ocr_and_math(self.drawing_engine.canvas.copy())
        else:
            self.status_bar.showMessage("Canvas is empty — nothing to solve.", 3000)

    def _on_text_recognized(self, text):
        """Handle OCR result from AI worker."""
        self._last_ocr_text = text
        self.ai_results.clear()
        self.ai_results.append(f"📝 <b>Recognized Text:</b>")
        self.ai_results.append(f"<pre>{text}</pre>")

        # Clear selection after reading
        self.drawing_engine.selection_active = False

    def _on_math_solved(self, result):
        """Handle math result from AI worker."""
        self.ai_results.append(f"\n🧮 <b>Math Result:</b>")
        if result.success:
            self.ai_results.append(f"<pre>{result.expression} = {result.solution}</pre>")
            # Save to database
            self.database.save_equation(
                self.session_id, result.expression, result.solution
            )
        else:
            self.ai_results.append(f"<i>⚠️ {result.error}</i>")

    def _on_shape_detected(self, result):
        """Handle shape detection result from AI worker."""
        if result.shape_type != "unknown":
            self.ai_results.append(
                f"📐 Shape: <b>{result.shape_type}</b> "
                f"(confidence: {result.confidence:.0%})"
            )

    # =========================================================================
    #  Utility
    # =========================================================================

    def _update_fps(self, frame_start):
        """Calculate and display FPS."""
        now = time.time()
        self._frame_times.append(now)
        # Keep only last 30 frame times
        self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) > 1:
            self._fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])
        self.fps_label.setText(f"FPS: {self._fps:.0f}")

    def _create_separator(self):
        """Create a horizontal line separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2a2a3a;")
        return sep

    def _create_vseparator(self):
        """Create a vertical line separator for the status bar."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(2)
        sep.setStyleSheet("color: #3a3a4a;")
        return sep

    def _show_about(self):
        """Show the About dialog."""
        QMessageBox.about(
            self,
            "About AI Smart Air Notebook",
            "<h2>AI Smart Air Notebook</h2>"
            "<p>Version 2.0 — Phase 2 (AI Intelligence)</p>"
            "<p>An intelligent touchless note-taking system using "
            "hand gestures, computer vision, and AI.</p>"
            "<p><b>Technologies:</b> MediaPipe, OpenCV, PyQt6, EasyOCR, SymPy, SQLite</p>"
            "<p><b>AI Features:</b> Shape Recognition, OCR, Math Solver</p>"
        )

    def _show_gesture_guide(self):
        """Show the gesture guide dialog."""
        QMessageBox.information(
            self,
            "Gesture Guide",
            "<h3>✋ Gesture Controls</h3>"
            "<table>"
            "<tr><td><b>☝️ Index Finger</b></td><td>Draw Mode</td></tr>"
            "<tr><td><b>✌️ Index + Middle</b></td><td>Selection Mode (Header: color/size, Canvas: area select)</td></tr>"
            "<tr><td><b>🖐️ Open Palm (hold 1.5s)</b></td><td>Clear Canvas</td></tr>"
            "<tr><td><b>🖕 Middle Finger Only</b></td><td>Eraser</td></tr>"
            "<tr><td><b>🤙 Pinky Only (hold 1s)</b></td><td>Undo</td></tr>"
            "<tr><td><b>🤘 Index + Pinky (hold 2s)</b></td><td>Save Canvas</td></tr>"
            "</table>"
            "<br><h4>✏️ How Drawing Works</h4>"
            "<ol>"
            "<li>Raise index finger → enters Draw Mode (pen UP)</li>"
            "<li><b>Hold still 1s</b> → 🟢 green ring fills → pen goes <b>DOWN</b></li>"
            "<li>Move finger → draws on canvas</li>"
            "<li><b>Hold still 1s</b> → 🔴 red ring fills → pen goes <b>UP</b></li>"
            "<li>Move to next letter position (no lines drawn)</li>"
            "<li>Hold still 1s again → pen DOWN, start next letter</li>"
            "</ol>"
            "<br><h4>🧠 AI Features</h4>"
            "<ul>"
            "<li><b>Shape Auto-Snap:</b> Toggle ON → rough circles/rectangles auto-snap to perfect shapes</li>"
            "<li><b>Read Text:</b> Select an area (✌️ below header), then click 'Read Text' for OCR</li>"
            "<li><b>Solve Math:</b> Write an equation, Read Text, then click 'Solve Math'</li>"
            "</ul>"
            "<br><p><b>💾 Save:</b> Use the Save button, <b>Ctrl+S</b>, or 🤘 gesture</p>"
            "<br><p><i>Tip: Use Selection mode below the header to select canvas areas for AI!</i></p>"
        )

    # =========================================================================
    #  Stylesheet
    # =========================================================================

    def _apply_stylesheet(self):
        """Apply the dark theme stylesheet."""
        self.setStyleSheet("""
            /* === Main Window === */
            QMainWindow {
                background-color: #0d0d14;
                color: #e0e0e8;
            }

            /* === Menu Bar === */
            QMenuBar {
                background-color: #12121c;
                color: #b0b0c0;
                border-bottom: 1px solid #1e1e2e;
                padding: 4px 0;
                font-size: 13px;
            }
            QMenuBar::item:selected {
                background-color: #1e1e30;
                border-radius: 4px;
            }
            QMenu {
                background-color: #16162a;
                color: #d0d0e0;
                border: 1px solid #2a2a40;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #2a2a50;
                border-radius: 4px;
            }

            /* === Tool Panel === */
            #toolPanel {
                background-color: #111120;
                border: 1px solid #1e1e30;
                border-radius: 16px;
            }
            #panelTitle {
                font-size: 18px;
                font-weight: bold;
                color: #00ffc8;
                padding: 8px 0;
            }

            /* === Canvas Container === */
            #canvasContainer {
                background-color: #0a0a0f;
                border: 1px solid #1e1e30;
                border-radius: 16px;
            }

            /* === Group Boxes === */
            QGroupBox {
                color: #8888a0;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #1e1e30;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 12px;
                color: #9090b0;
            }

            /* === Gesture Label === */
            #gestureLabel {
                font-size: 15px;
                font-weight: bold;
                color: #00ffc8;
                padding: 8px;
            }

            /* === Hold Progress Bar === */
            #holdProgress {
                background-color: #1a1a2e;
                border: none;
                border-radius: 4px;
            }
            #holdProgress::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00ffc8, stop: 1 #00aaff
                );
                border-radius: 4px;
            }

            /* === Buttons === */
            QPushButton {
                background-color: #1a1a2e;
                color: #d0d0e0;
                border: 1px solid #2a2a40;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #242440;
                border: 1px solid #3a3a60;
            }
            QPushButton:pressed {
                background-color: #2e2e50;
            }
            QPushButton:checked {
                background-color: #003322;
                border: 1px solid #00ffc8;
                color: #00ffc8;
            }

            #saveBtn {
                background-color: #0a2a20;
                border: 1px solid #00aa80;
                color: #00ffc8;
            }
            #saveBtn:hover {
                background-color: #0e3a2a;
            }

            #clearBtn {
                background-color: #2a0a0a;
                border: 1px solid #aa3030;
                color: #ff6060;
            }
            #clearBtn:hover {
                background-color: #3a1010;
            }

            /* === Slider === */
            QSlider::groove:horizontal {
                background: #1a1a2e;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ffc8;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00ffc8, stop: 1 #00aaff
                );
                border-radius: 3px;
            }

            /* === Status Bar === */
            QStatusBar {
                background-color: #0d0d14;
                color: #707088;
                border-top: 1px solid #1e1e2e;
                font-size: 12px;
                padding: 2px 8px;
            }
            #fpsLabel, #toolLabel, #sessionLabel {
                color: #8888a0;
                padding: 0 8px;
            }

            /* === Labels === */
            QLabel {
                color: #c0c0d0;
                font-size: 13px;
            }

            /* === AI Features Panel === */
            #autoSnapBtn {
                background-color: #1a1a2e;
                border: 1px solid #6a3aaa;
                color: #b080ff;
            }
            #autoSnapBtn:checked {
                background-color: #2a1a4a;
                border: 1px solid #9060ff;
                color: #c0a0ff;
            }
            #autoSnapBtn:hover {
                background-color: #2a2050;
            }

            #readTextBtn {
                background-color: #0a1a2a;
                border: 1px solid #2080cc;
                color: #60c0ff;
            }
            #readTextBtn:hover {
                background-color: #0e2a3a;
            }

            #solveMathBtn {
                background-color: #2a1a0a;
                border: 1px solid #cc8020;
                color: #ffb060;
            }
            #solveMathBtn:hover {
                background-color: #3a2a10;
            }

            #aiResults {
                background-color: #0a0a14;
                color: #c0c0d0;
                border: 1px solid #1e1e30;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)

    # =========================================================================
    #  Cleanup
    # =========================================================================

    def closeEvent(self, event):
        """Clean up resources on window close."""
        self._stop_camera()
        self.hand_tracker.release()
        self.database.close()
        event.accept()
