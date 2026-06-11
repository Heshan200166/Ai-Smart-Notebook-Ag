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
    QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon, QFont, QAction

from ui.canvas_widget import CanvasWidget
from modules.hand_tracker import HandTracker
from modules.gesture_controller import GestureController, Gesture
from modules.drawing_engine import DrawingEngine
from modules.database import NotebookDatabase


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

        self.eraser_btn = QPushButton("🧹  Eraser")
        self.eraser_btn.setObjectName("eraserBtn")
        self.eraser_btn.setCheckable(True)
        self.eraser_btn.setFixedHeight(40)
        self.eraser_btn.clicked.connect(self._on_eraser_toggled)
        tools_layout.addWidget(self.eraser_btn)

        self.undo_btn = QPushButton("↩️  Undo")
        self.undo_btn.setObjectName("undoBtn")
        self.undo_btn.setFixedHeight(40)
        self.undo_btn.clicked.connect(self._on_undo)
        tools_layout.addWidget(self.undo_btn)

        self.clear_btn = QPushButton("🗑️  Clear Canvas")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.clicked.connect(self._on_clear)
        tools_layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("💾  Save Drawing")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.setFixedHeight(40)
        self.save_btn.clicked.connect(self._on_save)
        tools_layout.addWidget(self.save_btn)

        tool_layout.addWidget(tools_group)

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
                # Check if finger is in the header UI area
                if pos[1] > self.HEADER_HEIGHT:
                    self.drawing_engine.draw(pos[0], pos[1])
        elif gesture == Gesture.ERASE:
            pos = self.hand_tracker.get_fingertip_position(2)  # Middle finger
            if pos and pos[1] > self.HEADER_HEIGHT:
                self.drawing_engine.set_eraser(True)
                self.drawing_engine.draw(pos[0], pos[1])
                self.drawing_engine.set_eraser(False)
        elif gesture == Gesture.SELECT:
            pos = self.hand_tracker.get_fingertip_position(1)  # Index finger
            if pos and pos[1] <= self.HEADER_HEIGHT:
                self._check_header_selection(pos[0], pos[1])
            self.drawing_engine.stop_drawing()
        else:
            self.drawing_engine.stop_drawing()

        # Handle debounced destructive gestures
        if self.gesture_controller.clear_triggered:
            self.drawing_engine.clear_canvas()
        if self.gesture_controller.save_triggered:
            self._on_save()

        # --- Step 4: Draw HUD on frame ---
        frame = self._draw_hud(frame, gesture)

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
        """Draw the on-screen header bar with color palette and tools."""
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

        # Brush size indicator
        size_x = self.COLOR_CIRCLE_START_X + len(DrawingEngine.COLOR_LIST) * self.COLOR_CIRCLE_SPACING + 30
        cv2.circle(frame, (size_x, self.COLOR_CIRCLE_Y),
                   self.drawing_engine.brush_size,
                   self.drawing_engine.color, -1, cv2.LINE_AA)
        cv2.putText(frame, "Size", (size_x - 15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Eraser indicator
        eraser_x = size_x + 70
        eraser_color = (0, 200, 255) if self.drawing_engine.eraser_mode else (100, 100, 100)
        cv2.rectangle(frame, (eraser_x - 25, 20), (eraser_x + 25, 55), eraser_color, -1)
        cv2.putText(frame, "ERA", (eraser_x - 18, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Gesture hint text (right side)
        hint = self.gesture_controller.get_gesture_display_name()
        text_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, hint, (w - text_size[0] - 20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

        return frame

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
            "<p>Version 1.0 — Phase 1</p>"
            "<p>An intelligent touchless note-taking system using "
            "hand gestures and computer vision.</p>"
            "<p><b>Technologies:</b> MediaPipe, OpenCV, PyQt6, SQLite</p>"
        )

    def _show_gesture_guide(self):
        """Show the gesture guide dialog."""
        QMessageBox.information(
            self,
            "Gesture Guide",
            "<h3>✋ Gesture Controls</h3>"
            "<table>"
            "<tr><td><b>☝️ Index Finger</b></td><td>Draw</td></tr>"
            "<tr><td><b>✌️ Index + Middle</b></td><td>Selection Mode</td></tr>"
            "<tr><td><b>🖐️ Open Palm (hold 1s)</b></td><td>Clear Canvas</td></tr>"
            "<tr><td><b>✊ Fist (hold 1s)</b></td><td>Save Drawing</td></tr>"
            "<tr><td><b>🤟 Thumb + Middle</b></td><td>Eraser</td></tr>"
            "</table>"
            "<br><p><i>Tip: Use Selection mode to pick colors from the header bar!</i></p>"
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
