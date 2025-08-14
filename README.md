# Body Tracker — Webcam Pose Scanner

Body Tracker is a lightweight desktop GUI that lets you scan your webcam(s) and visualize a live stream. It uses OpenCV for camera capture and can optionally use MediaPipe if present. As an alternative to MediaPipe, it can also use Ultralytics YOLOv8/YOLOv10 Pose for body keypoints. The app runs locally, requires no internet connection, and aims to be simple and robust for quick testing of camera streams and basic body-tracking scenarios.

Note: MediaPipe and Ultralytics are optional. If neither is installed, the app still runs as a basic webcam viewer.

## Features
- Enumerate connected cameras and select which one to open
- Start/stop live webcam preview
- Record video to local files in a 'saved-tracks' folder
- Status bar with basic runtime information
- Optional MediaPipe support (if mediapipe is installed), with on-camera pose skeleton lines (shoulders/hips/midline)
- Optional Ultralytics YOLO Pose support (if ultralytics is installed), automatically used when MediaPipe is unavailable
- Cross‑platform: Windows, macOS, Linux (depending on available camera backends)
- Simple, single‑file app

## Screenshots
Coming soon.

## Requirements
- Python 3.8+ (3.10+ recommended)
- OS support: Windows, macOS, Linux (you must have a supported camera backend)
- Permissions: allow the app to access the camera in your OS privacy settings

## Installation
This project doesn’t ship a requirements.txt. Install the base dependencies first, then optionally add MediaPipe and/or Ultralytics if your platform supports them.

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Base runtime dependencies (required)
pip install opencv-python numpy

# Optional: MediaPipe (pose overlay). If this fails, you can still run the app without it.
# Standard wheels (x86_64 Win/macOS/Linux, supported Python versions):
pip install mediapipe

# Apple Silicon macOS alternative (if standard mediapipe fails):
# pip install mediapipe-silicon

# Optional: Ultralytics YOLO Pose backend (alternative to MediaPipe)
# This installs the ultralytics package (will also install torch if needed)
pip install ultralytics
```

Notes:
- MediaPipe and Ultralytics are optional. If installation fails or wheels aren’t available for your platform/Python, just skip them — the app works as a basic webcam viewer.
- Tkinter usually ships with standard Python installers. If your Python build lacks Tkinter, install it via your OS package manager.
- On Linux you might need additional system packages for video backends (e.g., v4l2, gstreamer plugins).
- On Windows, the code may use the DirectShow flag (CAP_DSHOW) when available for more reliable enumeration.

## Usage
Run the app from the project root:

```bash
python main.py
```

Steps:
1. Click “Refresh” to enumerate available webcams.
2. Choose a camera from the dropdown.
3. Click “Start” to begin streaming; “Stop” to end.
4. (Optional) Click “Record” to start/stop saving the current stream into the 'saved-tracks' folder.
5. Use the “Help” button for quick guidance.

### YOLO Pose configuration (optional)
If Ultralytics is installed and MediaPipe is not, the app uses YOLO Pose automatically. You can tweak it via environment variables before launching:

- `YOLO_POSE_MODEL` (default: `yolov8n-pose.pt`) — model name or path. You can also use YOLOv10 pose models if available.
- `YOLO_IMG_SIZE` (default: `640`) — inference image size.
- `YOLO_CONF` (default: `0.25`) — confidence threshold.

Example:

```bash
YOLO_POSE_MODEL=yolov8s-pose.pt YOLO_IMG_SIZE=640 YOLO_CONF=0.25 python main.py
```

## Project Structure
```
.
├── main.py        # Tkinter GUI, camera enumeration, video processor thread
└── README.md      # This file
```

High‑level code components (see main.py):
- CameraEnumerator: scans indices and returns a list of working camera ids.
- VideoProcessor (thread): grabs frames via OpenCV; uses MediaPipe if available, otherwise uses Ultralytics YOLO Pose if available; emits frames to the UI as base64 PNGs.
- App (tk.Tk): builds the UI, manages start/stop, updates status, and displays frames.

## Troubleshooting
- OpenCV not found or not working:
  - Ensure `opencv-python` is installed and importable:
    - `python -c "import cv2; print(cv2.__version__)"`
  - On Linux, if import fails with errors mentioning GL/X11, install system OpenGL libs:
    - `sudo apt-get update && sudo apt-get install -y libgl1`
  - On servers or headless environments, prefer the headless wheel:
    - `pip install opencv-python-headless`
  - If camera won’t open, try other indices and ensure no other app is using the camera. On Linux, device paths look like `/dev/video0`, `/dev/video1`.
- MediaPipe/Ultralytics are optional: if neither is installed, the app still works as a viewer.
- “No matching distribution found for mediapipe”: this usually means your platform or Python version doesn’t have a prebuilt wheel.
  - Check your environment:
    - `python -c "import sys, platform; print(sys.version); print(platform.platform()); print(platform.machine())"`
  - Common fixes:
    - Ensure a supported Python version (3.8–3.12 tends to be supported depending on platform; check the mediapipe PyPI page for current info).
    - On Apple Silicon macOS, try: `pip install mediapipe-silicon`.
    - On Linux with older glibc or on ARM/aarch64 SBCs, official wheels may be unavailable; consider upgrading OS/Python or running without MediaPipe.
    - Try pinning a specific version, e.g.: `pip install "mediapipe<0.11"` (adjust as per PyPI availability).
  - If it still fails, just skip MediaPipe; the viewer will run without pose overlays.
- No cameras detected: verify OS permissions, test with another app (e.g., system camera app), try different indices, and click Refresh.
- Linux backends: you may need to install or enable V4L2/GStreamer. On headless servers, you won’t have a GUI; consider running with a desktop session.
- macOS permissions: grant Terminal or your Python interpreter camera access via System Settings → Privacy & Security → Camera.

## Development
Suggested workflow:
- Use a virtual environment.
- Keep dependencies minimal.
- Prefer small, incremental changes and test on your platform.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
