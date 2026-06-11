# AI Smart Air Notebook 🖐️✨

An intelligent touchless note-taking system that enables users to write, draw, and interact with digital content using hand gestures and voice commands.

## Features (Phase 1)

- **✋ Real-time Hand Tracking** — MediaPipe-powered 21-landmark detection
- **✏️ Air Drawing** — Draw in the air with your index finger
- **🎨 Color Palette** — 8 vibrant colors, selectable via UI or gestures
- **🧹 Eraser & Undo** — Full drawing control
- **🤟 Gesture Controls** — Draw, Select, Clear, Save, Erase
- **💾 Session Management** — SQLite-backed sessions with PNG export
- **🖥️ Modern Desktop UI** — Dark-themed PyQt6 interface

## Gesture Guide

| Gesture | Action |
|---------|--------|
| ☝️ Index Finger | Draw |
| ✌️ Index + Middle | Selection Mode |
| 🖐️ Open Palm (hold 1s) | Clear Canvas |
| ✊ Fist (hold 1s) | Save Drawing |
| 🤟 Thumb + Middle | Eraser |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| MediaPipe | Hand landmark detection |
| OpenCV | Camera capture & image processing |
| NumPy | Coordinate math |
| PyQt6 | Desktop UI framework |
| SQLite | Session & metadata storage |

## Project Structure

```
AI_Smart_Notebook/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── modules/
│   ├── hand_tracker.py        # MediaPipe hand tracking
│   ├── gesture_controller.py  # Gesture interpretation
│   ├── drawing_engine.py      # Canvas & drawing logic
│   └── database.py            # SQLite storage
├── ui/
│   ├── canvas_widget.py       # OpenCV frame display
│   └── main_window.py         # Main application window
├── sessions/                  # Saved drawings
├── database/                  # SQLite database
└── exports/                   # Exported files
```

## Roadmap

- [x] Phase 1: Hand Tracking + Air Drawing + Gestures + UI
- [ ] Phase 2: Voice Notes (Whisper)
- [ ] Phase 3: OCR Integration (EasyOCR)
- [ ] Phase 4: AI Math Solver (SymPy)
- [ ] Phase 5: Smart Notebook (Search, Export, History)

## License

MIT