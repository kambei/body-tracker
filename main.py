import sys
import threading
import time
from typing import List, Optional

# GUI
import tkinter as tk
from tkinter import ttk, messagebox

import os
# Reduce OpenCV console spam before import (must be set pre-import)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
# Avoid probing the OBSENSOR (RealSense) backend which can spam errors like
# "obsensor_uvc_stream_channel.cpp: getStreamChannelGroup Camera index out of range"
# These environment flags are safe no-ops if the backend isn't present.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("OPENCV_VIDEOIO_DISABLE_OBSENSOR", "1")

# Try to import OpenCV and MediaPipe
CV2_IMPORT_ERROR = None
try:
    import cv2
    try:
        # Reduce noisy backend warnings (e.g., V4L2/GStreamer open failures during probing)
        from cv2.utils import logging as cv2_logging
        cv2_logging.setLogLevel(cv2_logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = str(e)

try:
    import mediapipe as mp
except Exception:
    mp = None

# Try to import Ultralytics YOLO (pose)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import base64
import numpy as np
import glob
from datetime import datetime


def open_capture(index: int):
    """Open a cv2.VideoCapture with platform-appropriate backend fallbacks.
    Returns an opened cv2.VideoCapture or None.
    """
    if cv2 is None:
        return None
    # Backend preference per OS
    if sys.platform.startswith("win"):
        backend_ids = [getattr(cv2, 'CAP_DSHOW', 0), getattr(cv2, 'CAP_MSMF', 0)]
    elif sys.platform == "darwin":
        backend_ids = [getattr(cv2, 'CAP_AVFOUNDATION', 0), getattr(cv2, 'CAP_QT', 0)]
    else:  # Linux and others
        backend_ids = [getattr(cv2, 'CAP_V4L2', 0), getattr(cv2, 'CAP_GSTREAMER', 0)]

    # Remove unknown/zero backends to avoid CAP_ANY auto-probing
    backend_ids = [be for be in backend_ids if isinstance(be, int) and be != 0]

    for be in backend_ids:
        cap = None
        try:
            cap = cv2.VideoCapture(index, be)
            if cap and cap.isOpened():
                return cap
        except Exception:
            pass
        finally:
            if cap and not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
    # Linux-specific fallback: try device path directly, e.g. /dev/video0
    if os.name == 'posix' and sys.platform.startswith('linux'):
        dev_path = f"/dev/video{index}"
        if os.path.exists(dev_path):
            try:
                cap = cv2.VideoCapture(dev_path)
                if cap and cap.isOpened():
                    return cap
                if cap:
                    cap.release()
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
    return None

APP_TITLE = "Body Tracker - Webcam Pose Scanner"


class CameraEnumerator:
    def __init__(self, max_index: int = 5, timeout_sec: float = 0.5):
        self.max_index = max_index
        self.timeout_sec = timeout_sec

    def list_cameras(self) -> List[int]:
        if cv2 is None:
            return []
        cams = []
        # On Linux, prefer enumerating actual device nodes to avoid probing non-existent indices
        candidate_indices: List[int] = []
        try:
            if os.name == 'posix' and sys.platform.startswith('linux'):
                devs = sorted(glob.glob('/dev/video*'))
                for d in devs:
                    # Extract index from basename 'videoN' or 'videoNN'
                    try:
                        base = os.path.basename(d)
                        if base.startswith('video'):
                            idx = int(base[5:])
                        else:
                            continue
                    except Exception:
                        idx = None
                    if idx is not None and idx not in candidate_indices:
                        candidate_indices.append(idx)
        except Exception:
            pass
        # Fallback if none detected
        if not candidate_indices:
            candidate_indices = list(range(self.max_index + 1))

        for idx in candidate_indices:
            cap = open_capture(idx)
            ok = False
            if cap and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    # Try to grab one frame quickly
                    start = time.time()
                    while time.time() - start < self.timeout_sec:
                        ret, _ = cap.read()
                        if ret:
                            ok = True
                            break
                        time.sleep(0.05)
                except Exception:
                    ok = False
            if ok:
                cams.append(idx)
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
        return cams


class VideoProcessor(threading.Thread):
    def __init__(self, cam_index: int, frame_callback, status_callback, stop_event: threading.Event, *, weight_kg: float = 70.0, activity_met: float = 3.0, overlay_calories: bool = True, active_seconds_provider=None):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.frame_callback = frame_callback
        self.status_callback = status_callback
        self.stop_event = stop_event
        self.cap: Optional[cv2.VideoCapture] = None
        self.pose = None
        self.drawer = None
        # Recording-related
        self._record_lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording: bool = False
        self._record_path: Optional[str] = None
        # Motion detection (frame-diff based)
        self._prev_gray = None
        self._frames_in_sec = 0
        self._motion_frames_in_sec = 0
        # Tuneable thresholds via env vars
        try:
            self._motion_pixel_thresh = int(os.environ.get("BT_MOTION_DIFF_THRESH", "15"))
        except Exception:
            self._motion_pixel_thresh = 15
        try:
            self._motion_ratio_thresh = float(os.environ.get("BT_MOTION_RATIO_THRESH", "0.005"))  # 0.5% pixels
        except Exception:
            self._motion_ratio_thresh = 0.005
        try:
            self._motion_frames_ratio = float(os.environ.get("BT_MOTION_FRAMES_RATIO", "0.1"))  # 10% of frames in the second
        except Exception:
            self._motion_frames_ratio = 0.1
        # Full-body gating configuration
        try:
            self._motion_mode = os.environ.get("BT_MOTION_MODE", "full").strip().lower()
        except Exception:
            self._motion_mode = "full"
        try:
            self._body_pix_thresh = float(os.environ.get("BT_BODY_PIX_THRESH", "8"))
        except Exception:
            self._body_pix_thresh = 8.0
        # Per-second accumulators for landmark displacements
        self._upper_disp_sum = 0.0
        self._lower_disp_sum = 0.0
        self._body_disp_frames = 0
        # Previous keypoints for displacement calc
        self._prev_keypoints = None
        # Calorie overlay/session
        self.weight_kg = float(weight_kg)
        self.activity_met = float(activity_met)
        self.overlay_calories = bool(overlay_calories)
        self.session_start_ts = time.time()
        self.session_active_seconds = 0
        self._last_move_increment_sec = None
        self._active_seconds_provider = active_seconds_provider

    def start_recording(self, path: str):
        """Start writing frames to the specified video file path.
        Can be called from UI thread while run() is looping.
        """
        if cv2 is None:
            return False
        with self._record_lock:
            if self._is_recording:
                return True
            if not self.cap or not self.cap.isOpened():
                return False
            # Ensure destination directory exists
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            except Exception:
                pass
            # Determine frame size and fps (properties may be 0 on some backends)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 640
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or 480
            fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 30.0
            # Try multiple codecs for robustness across platforms
            ext = os.path.splitext(path)[1].lower()
            if ext == ".mp4":
                codec_candidates = ["mp4v", "avc1", "H264", "X264"]
            else:
                codec_candidates = ["XVID", "MJPG", "DIVX"]
            writer = None
            for codec in codec_candidates:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    w = cv2.VideoWriter(path, fourcc, fps, (width, height))
                    if w and w.isOpened():
                        writer = w
                        break
                    else:
                        if w:
                            try:
                                w.release()
                            except Exception:
                                pass
                except Exception:
                    try:
                        w.release()
                    except Exception:
                        pass
                    continue
            if writer is None:
                return False
            self._writer = writer
            self._is_recording = True
            self._record_path = path
            self.status_callback(f"Recording started: {os.path.basename(path)}")
            return True

    def stop_recording(self):
        """Stop recording if active."""
        with self._record_lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:
                    pass
            was_recording = self._is_recording
            saved_path = self._record_path
            self._writer = None
            self._is_recording = False
            self._record_path = None
        if was_recording and saved_path:
            try:
                full = os.path.abspath(saved_path)
            except Exception:
                full = saved_path
            self.status_callback(f"Recording saved to: {full}")

    def run(self):
        if cv2 is None:
            details = f" | {CV2_IMPORT_ERROR}" if CV2_IMPORT_ERROR else ""
            self.status_callback("Error: OpenCV (cv2) is not available" + details + ". Try: pip install opencv-python (or opencv-python-headless on servers); on Linux: sudo apt-get install -y libgl1")
            return
        # Backends: prefer MediaPipe if available, else Ultralytics YOLO Pose; else no overlays
        use_mediapipe = mp is not None
        use_yolo = (YOLO is not None)
        try:
            self.cap = open_capture(self.cam_index)
            if not self.cap or not self.cap.isOpened():
                self.status_callback(f"Error: Cannot open camera {self.cam_index}. Try a different index or check OS permissions.")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            last_ts = time.time()
            frame_count = 0

            # Helper to draw calories overlay onto BGR frame
            def _draw_cal_overlay(img_bgr):
                if not self.overlay_calories:
                    return img_bgr
                try:
                    h, w = img_bgr.shape[:2]
                    elapsed = max(0, int(time.time() - self.session_start_ts))
                    mm = (elapsed % 3600) // 60
                    ss = elapsed % 60
                    # Use app-provided active seconds if available for consistency with status bar
                    if callable(self._active_seconds_provider):
                        try:
                            active_sec = max(0, int(self._active_seconds_provider()))
                        except Exception:
                            active_sec = max(0, int(self.session_active_seconds))
                    else:
                        active_sec = max(0, int(self.session_active_seconds))
                    kcal = self.activity_met * self.weight_kg * (active_sec / 3600.0)
                    text = f"Elapsed {mm:02d}:{ss:02d} | {kcal:.1f} kcal"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.7
                    thickness = 2
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    x, y = 10, 10 + th
                    # background box
                    cv2.rectangle(img_bgr, (x-8, y-th-8), (x-8+tw+16, y+8), (0, 0, 0), thickness=-1)
                    # text with slight shadow
                    cv2.putText(img_bgr, text, (x+1, y+1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
                    cv2.putText(img_bgr, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                except Exception:
                    pass
                return img_bgr

            if use_mediapipe:
                mp_pose = mp.solutions.pose
                mp_holistic = mp.solutions.holistic
                mp_drawing = mp.solutions.drawing_utils
                mp_styles = mp.solutions.drawing_styles
                self.status_callback(f"Streaming from camera {self.cam_index} with MediaPipe (pose+face+hands)...")
                with mp_holistic.Holistic(model_complexity=1, smooth_landmarks=True, refine_face_landmarks=True) as holistic:
                    while not self.stop_event.is_set():
                        ret, frame = self.cap.read()
                        if not ret:
                            self.status_callback("Warning: Failed to read from camera.")
                            time.sleep(0.05)
                            continue

                        # Convert BGR to RGB for MediaPipe
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb.flags.writeable = False
                        results = holistic.process(rgb)
                        rgb.flags.writeable = True

                        # Draw landmarks on an RGB image we will display
                        disp = rgb.copy()
                        if results.pose_landmarks is not None:
                            # Draw default MediaPipe pose landmarks and connections (skeleton)
                            mp_drawing.draw_landmarks(
                                disp,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )

                            # Add prominent guide lines for body tracker overlay (shoulders, hips, midline)
                            try:
                                h, w = disp.shape[:2]
                                lm = results.pose_landmarks.landmark
                                # Landmark indices (MediaPipe Pose):
                                # 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
                                ls = lm[11]
                                rs = lm[12]
                                lh = lm[23]
                                rh = lm[24]
                                # Convert normalized [0,1] to pixel coordinates
                                ls_xy = (int(ls.x * w), int(ls.y * h))
                                rs_xy = (int(rs.x * w), int(rs.y * h))
                                lh_xy = (int(lh.x * w), int(lh.y * h))
                                rh_xy = (int(rh.x * w), int(rh.y * h))

                                # Shoulder line (red)
                                cv2.line(disp, ls_xy, rs_xy, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                                # Hip line (red)
                                cv2.line(disp, lh_xy, rh_xy, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                                # Mid points
                                mid_shoulder = ((ls_xy[0] + rs_xy[0]) // 2, (ls_xy[1] + rs_xy[1]) // 2)
                                mid_hip = ((lh_xy[0] + rh_xy[0]) // 2, (lh_xy[1] + rh_xy[1]) // 2)
                                # Vertical midline (approx spine) (red)
                                cv2.line(disp, mid_shoulder, mid_hip, (0, 0, 255), 4, lineType=cv2.LINE_AA)

                                # Small circles for keypoints to make lines clearer (red)
                                for pt in (ls_xy, rs_xy, lh_xy, rh_xy, mid_shoulder, mid_hip):
                                    cv2.circle(disp, pt, 5, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                                    cv2.circle(disp, pt, 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

                                # Update full-body displacement accumulators for this frame (MediaPipe)
                                try:
                                    kp = {'ls': ls_xy, 'rs': rs_xy, 'lh': lh_xy, 'rh': rh_xy}
                                    if self._prev_keypoints is not None:
                                        import math
                                        def dist(a, b):
                                            return math.hypot(float(a[0]-b[0]), float(a[1]-b[1]))
                                        upper = 0.5 * (dist(kp['ls'], self._prev_keypoints['ls']) + dist(kp['rs'], self._prev_keypoints['rs']))
                                        lower = 0.5 * (dist(kp['lh'], self._prev_keypoints['lh']) + dist(kp['rh'], self._prev_keypoints['rh']))
                                        self._upper_disp_sum += upper
                                        self._lower_disp_sum += lower
                                        self._body_disp_frames += 1
                                    self._prev_keypoints = kp
                                except Exception:
                                    pass
                            except Exception:
                                # If any index is missing or conversion fails, skip custom lines
                                pass

                        # Draw face mesh contours (head, eyes, mouth) if available
                        if getattr(results, 'face_landmarks', None) is not None:
                            try:
                                mp_drawing.draw_landmarks(
                                    disp,
                                    results.face_landmarks,
                                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=0)
                                )
                                # Optionally add a subtle tesselation to better show face shape
                                mp_drawing.draw_landmarks(
                                    disp,
                                    results.face_landmarks,
                                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=0)
                                )
                            except Exception:
                                pass

                        # Draw hand connections for both hands if available
                        try:
                            if getattr(results, 'left_hand_landmarks', None) is not None:
                                mp_drawing.draw_landmarks(
                                    disp,
                                    results.left_hand_landmarks,
                                    mp.solutions.hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                                )
                            if getattr(results, 'right_hand_landmarks', None) is not None:
                                mp_drawing.draw_landmarks(
                                    disp,
                                    results.right_hand_landmarks,
                                    mp.solutions.hands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                                )
                        except Exception:
                            pass

                        # Accentuate feet (ankle to toes) if pose landmarks present
                        if getattr(results, 'pose_landmarks', None) is not None:
                            try:
                                h, w = disp.shape[:2]
                                lm = results.pose_landmarks.landmark
                                # Left leg: knee(25)->ankle(27)->heel(29)->foot_index(31)
                                pts_left = [25, 27, 29, 31]
                                # Right leg: knee(26)->ankle(28)->heel(30)->foot_index(32)
                                pts_right = [26, 28, 30, 32]
                                def to_xy(idx):
                                    p = lm[idx]
                                    return (int(p.x * w), int(p.y * h))
                                for seq, color in ((pts_left, (0, 0, 255)), (pts_right, (0, 0, 255))):
                                    for a, b in zip(seq, seq[1:]):
                                        cv2.line(disp, to_xy(a), to_xy(b), color, 3, lineType=cv2.LINE_AA)
                            except Exception:
                                pass

                        # Convert to Tkinter-compatible PNG via memory (avoid PIL dependency)
                        try:
                            bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
                            # Calories overlay
                            bgr = _draw_cal_overlay(bgr)
                            # Motion detection update (use displayed BGR)
                            try:
                                small = cv2.resize(bgr, (320, 180))
                                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                                if self._prev_gray is not None:
                                    diff = cv2.absdiff(gray, self._prev_gray)
                                    _, thr = cv2.threshold(diff, self._motion_pixel_thresh, 255, cv2.THRESH_BINARY)
                                    motion_ratio = float(np.count_nonzero(thr)) / float(thr.size)
                                    if motion_ratio > self._motion_ratio_thresh:
                                        self._motion_frames_in_sec += 1
                                self._frames_in_sec += 1
                                self._prev_gray = gray
                            except Exception:
                                pass
                            # Write to video file if recording
                            with self._record_lock:
                                if self._is_recording and self._writer is not None:
                                    try:
                                        self._writer.write(bgr)
                                    except Exception:
                                        pass
                            success, buf = cv2.imencode('.png', bgr)
                            if success:
                                png_bytes = buf.tobytes()
                                b64 = base64.b64encode(png_bytes)
                                self.frame_callback(b64)
                        except Exception:
                            pass

                        frame_count += 1
                        now = time.time()
                        if now - last_ts >= 1.0:
                            fps = frame_count / (now - last_ts)
                            try:
                                moving_second = (self._motion_frames_in_sec / max(1, self._frames_in_sec)) > self._motion_frames_ratio
                            except Exception:
                                moving_second = False
                            # Optional full-body gate (requires both upper and lower body displacement)
                            try:
                                if self._motion_mode in ("full", "strict") and self._body_disp_frames > 0:
                                    upper_avg = self._upper_disp_sum / max(1, self._body_disp_frames)
                                    lower_avg = self._lower_disp_sum / max(1, self._body_disp_frames)
                                    body_ok = (upper_avg >= self._body_pix_thresh) and (lower_avg >= self._body_pix_thresh)
                                    moving_second = moving_second and body_ok
                            except Exception:
                                pass
                            self.status_callback(f"Streaming from camera {self.cam_index} - {fps:.1f} FPS | MOVE={'1' if moving_second else '0'}")
                            # reset second counters
                            self._frames_in_sec = 0
                            self._motion_frames_in_sec = 0
                            self._upper_disp_sum = 0.0
                            self._lower_disp_sum = 0.0
                            self._body_disp_frames = 0
                            last_ts = now
                            frame_count = 0
            elif use_yolo:
                # Ultralytics YOLO Pose backend (fallback/alternative to MediaPipe)
                model_name = os.environ.get("YOLO_POSE_MODEL", "yolov8n-pose.pt")
                try:
                    model = YOLO(model_name)
                    self.status_callback(f"Streaming from camera {self.cam_index} with YOLO Pose ({model_name})...")
                except Exception as e:
                    self.status_callback(f"Error loading YOLO model '{model_name}': {e}. Falling back to raw stream.")
                    use_yolo = False
                while not self.stop_event.is_set() and use_yolo:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.status_callback("Warning: Failed to read from camera.")
                        time.sleep(0.05)
                        continue
                    try:
                        # Run pose inference (BGR frame). Adjust imgsz/conf if needed via env vars
                        imgsz = int(os.environ.get("YOLO_IMG_SIZE", "640"))
                        conf = float(os.environ.get("YOLO_CONF", "0.25"))
                        results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
                        res0 = results[0]
                        # Get plotted image with skeletons (BGR)
                        disp_bgr = res0.plot()

                        # Optional extra guide lines for the first detected person (shoulders/hips/midline)
                        try:
                            if hasattr(res0, 'keypoints') and res0.keypoints is not None and len(res0.keypoints) > 0:
                                kpts = res0.keypoints.xy  # tensor [N,17,2]
                                if kpts is not None and len(kpts) > 0:
                                    pts = kpts[0]  # first person
                                    # COCO indices: 5-LShoulder, 6-RShoulder, 11-LHip, 12-RHip
                                    ls_xy = tuple(map(int, pts[5].tolist()))
                                    rs_xy = tuple(map(int, pts[6].tolist()))
                                    lh_xy = tuple(map(int, pts[11].tolist()))
                                    rh_xy = tuple(map(int, pts[12].tolist()))
                                    # Draw lines in red (BGR (0,0,255))
                                    cv2.line(disp_bgr, ls_xy, rs_xy, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                                    cv2.line(disp_bgr, lh_xy, rh_xy, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                                    mid_shoulder = ((ls_xy[0] + rs_xy[0]) // 2, (ls_xy[1] + rs_xy[1]) // 2)
                                    mid_hip = ((lh_xy[0] + rh_xy[0]) // 2, (lh_xy[1] + rh_xy[1]) // 2)
                                    cv2.line(disp_bgr, mid_shoulder, mid_hip, (0, 0, 255), 4, lineType=cv2.LINE_AA)
                                    for pt in (ls_xy, rs_xy, lh_xy, rh_xy, mid_shoulder, mid_hip):
                                        cv2.circle(disp_bgr, pt, 5, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
                                    # Update full-body displacement accumulators for this frame (YOLO)
                                    try:
                                        kp = {'ls': ls_xy, 'rs': rs_xy, 'lh': lh_xy, 'rh': rh_xy}
                                        if self._prev_keypoints is not None:
                                            import math
                                            def dist(a, b):
                                                return math.hypot(float(a[0]-b[0]), float(a[1]-b[1]))
                                            upper = 0.5 * (dist(kp['ls'], self._prev_keypoints['ls']) + dist(kp['rs'], self._prev_keypoints['rs']))
                                            lower = 0.5 * (dist(kp['lh'], self._prev_keypoints['lh']) + dist(kp['rh'], self._prev_keypoints['rh']))
                                            self._upper_disp_sum += upper
                                            self._lower_disp_sum += lower
                                            self._body_disp_frames += 1
                                        self._prev_keypoints = kp
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # Calories overlay
                        disp_bgr = _draw_cal_overlay(disp_bgr)
                        # Motion detection update (use displayed BGR)
                        try:
                            small = cv2.resize(disp_bgr, (320, 180))
                            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                            if self._prev_gray is not None:
                                diff = cv2.absdiff(gray, self._prev_gray)
                                _, thr = cv2.threshold(diff, self._motion_pixel_thresh, 255, cv2.THRESH_BINARY)
                                motion_ratio = float(np.count_nonzero(thr)) / float(thr.size)
                                if motion_ratio > self._motion_ratio_thresh:
                                    self._motion_frames_in_sec += 1
                            self._frames_in_sec += 1
                            self._prev_gray = gray
                        except Exception:
                            pass
                        # Write to video if recording and send to UI as PNG
                        with self._record_lock:
                            if self._is_recording and self._writer is not None:
                                try:
                                    self._writer.write(disp_bgr)
                                except Exception:
                                    pass
                        success, buf = cv2.imencode('.png', disp_bgr)
                        if success:
                            png_bytes = buf.tobytes()
                            b64 = base64.b64encode(png_bytes)
                            self.frame_callback(b64)
                    except Exception:
                        # If YOLO processing fails, do a simple pass-through for this frame
                        try:
                            # Update motion on raw frame as fallback
                            try:
                                small = cv2.resize(frame, (320, 180))
                                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                                if self._prev_gray is not None:
                                    diff = cv2.absdiff(gray, self._prev_gray)
                                    _, thr = cv2.threshold(diff, self._motion_pixel_thresh, 255, cv2.THRESH_BINARY)
                                    motion_ratio = float(np.count_nonzero(thr)) / float(thr.size)
                                    if motion_ratio > self._motion_ratio_thresh:
                                        self._motion_frames_in_sec += 1
                                self._frames_in_sec += 1
                                self._prev_gray = gray
                            except Exception:
                                pass
                            success, buf = cv2.imencode('.png', frame)
                            if success:
                                png_bytes = buf.tobytes()
                                b64 = base64.b64encode(png_bytes)
                                self.frame_callback(b64)
                        except Exception:
                            pass

                    frame_count += 1
                    now = time.time()
                    if now - last_ts >= 1.0:
                        fps = frame_count / (now - last_ts)
                        try:
                            moving_second = (self._motion_frames_in_sec / max(1, self._frames_in_sec)) > self._motion_frames_ratio
                        except Exception:
                            moving_second = False
                        # Optional full-body gate (requires both upper and lower body displacement)
                        try:
                            if self._motion_mode in ("full", "strict") and self._body_disp_frames > 0:
                                upper_avg = self._upper_disp_sum / max(1, self._body_disp_frames)
                                lower_avg = self._lower_disp_sum / max(1, self._body_disp_frames)
                                body_ok = (upper_avg >= self._body_pix_thresh) and (lower_avg >= self._body_pix_thresh)
                                moving_second = moving_second and body_ok
                        except Exception:
                            pass
                        self.status_callback(f"Streaming from camera {self.cam_index} - {fps:.1f} FPS | MOVE={'1' if moving_second else '0'}")
                        # reset second counters
                        self._frames_in_sec = 0
                        self._motion_frames_in_sec = 0
                        self._upper_disp_sum = 0.0
                        self._lower_disp_sum = 0.0
                        self._body_disp_frames = 0
                        last_ts = now
                        frame_count = 0
            else:
                self.status_callback(f"Streaming from camera {self.cam_index} (no MediaPipe)...")
                while not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret:
                        self.status_callback("Warning: Failed to read from camera.")
                        time.sleep(0.05)
                        continue

                    try:
                        # Motion detection update (use original BGR frame)
                        try:
                            small = cv2.resize(frame, (320, 180))
                            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                            if self._prev_gray is not None:
                                diff = cv2.absdiff(gray, self._prev_gray)
                                _, thr = cv2.threshold(diff, self._motion_pixel_thresh, 255, cv2.THRESH_BINARY)
                                motion_ratio = float(np.count_nonzero(thr)) / float(thr.size)
                                if motion_ratio > self._motion_ratio_thresh:
                                    self._motion_frames_in_sec += 1
                            self._frames_in_sec += 1
                            self._prev_gray = gray
                        except Exception:
                            pass
                        # Overlay calories on the BGR frame
                        frame = _draw_cal_overlay(frame)
                        # Write to video file if recording using the overlaid BGR frame
                        with self._record_lock:
                            if self._is_recording and self._writer is not None:
                                try:
                                    self._writer.write(frame)
                                except Exception:
                                    pass
                        success, buf = cv2.imencode('.png', frame)
                        if success:
                            png_bytes = buf.tobytes()
                            b64 = base64.b64encode(png_bytes)
                            self.frame_callback(b64)
                    except Exception:
                        pass

                    frame_count += 1
                    now = time.time()
                    if now - last_ts >= 1.0:
                        fps = frame_count / (now - last_ts)
                        try:
                            moving_second = (self._motion_frames_in_sec / max(1, self._frames_in_sec)) > self._motion_frames_ratio
                        except Exception:
                            moving_second = False
                        # Optional full-body gate (requires both upper and lower body displacement)
                        try:
                            if self._motion_mode in ("full", "strict") and self._body_disp_frames > 0:
                                upper_avg = self._upper_disp_sum / max(1, self._body_disp_frames)
                                lower_avg = self._lower_disp_sum / max(1, self._body_disp_frames)
                                body_ok = (upper_avg >= self._body_pix_thresh) and (lower_avg >= self._body_pix_thresh)
                                moving_second = moving_second and body_ok
                        except Exception:
                            pass
                        self.status_callback(f"Streaming from camera {self.cam_index} - {fps:.1f} FPS | MOVE={'1' if moving_second else '0'}")
                        # reset second counters
                        self._frames_in_sec = 0
                        self._motion_frames_in_sec = 0
                        self._upper_disp_sum = 0.0
                        self._lower_disp_sum = 0.0
                        self._body_disp_frames = 0
                        last_ts = now
                        frame_count = 0
        finally:
            # Ensure recording is properly closed
            try:
                self.stop_recording()
            except Exception:
                pass
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_label = None
        self.status_var = tk.StringVar(value="Idle")

        self.processor: Optional[VideoProcessor] = None
        self.stop_event = threading.Event()
        self.tk_image = None  # keep reference to prevent GC
        # Recording state
        self.is_recording = False
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_dir = os.path.join(base_dir, "saved-tracks")
        self.current_record_path: Optional[str] = None

        # Session & calorie tracking
        try:
            self.user_weight_kg = float(os.environ.get("BT_WEIGHT_KG", "70"))
        except Exception:
            self.user_weight_kg = 70.0
        try:
            self.activity_met = float(os.environ.get("BT_ACTIVITY_MET", "3.0"))  # light activity default
        except Exception:
            self.activity_met = 3.0
        self.overlay_calories: bool = True
        self.session_start_ts: Optional[float] = None
        self.session_active_seconds: int = 0
        self._last_move_increment_sec: Optional[int] = None

        self._build_ui()
        # Populate cameras after UI loads
        self.after(100, self.refresh_cameras)

    def _build_ui(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Camera:").pack(side=tk.LEFT)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(controls, textvariable=self.cam_var, state="readonly", width=30)
        self.cam_combo.pack(side=tk.LEFT, padx=6)

        self.refresh_btn = ttk.Button(controls, text="Refresh", command=self.refresh_cameras)
        self.refresh_btn.pack(side=tk.LEFT)

        self.start_btn = ttk.Button(controls, text="Start", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=6)

        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        self.help_btn = ttk.Button(controls, text="Help", command=self.show_help)
        self.help_btn.pack(side=tk.RIGHT)

        self.record_btn = ttk.Button(controls, text="Record", command=self.toggle_record, state=tk.DISABLED)
        self.record_btn.pack(side=tk.RIGHT, padx=6)

        self.video_label = ttk.Label(container)
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=(8, 4))

        status_bar = ttk.Label(container, textvariable=self.status_var, anchor="w")
        status_bar.pack(fill=tk.X)

        # Menu bar with Settings
        try:
            menubar = tk.Menu(self)
            settings_menu = tk.Menu(menubar, tearoff=0)
            settings_menu.add_command(label="Settings...", command=self.show_settings)
            menubar.add_cascade(label="Settings", menu=settings_menu)
            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label="Help", command=self.show_help)
            menubar.add_cascade(label="Help", menu=help_menu)
            self.config(menu=menubar)
        except Exception:
            pass

        # Size
        self.minsize(800, 600)

    def set_status(self, text: str):
        # Called from worker thread via thread-safe StringVar set using after
        def _parse_move_and_maybe_increment():
            if self.session_start_ts is None:
                return
            try:
                idx = text.rfind("MOVE=")
                if idx != -1:
                    val = text[idx+5:idx+6]
                    if val == '1':
                        cur_sec = int(time.time())
                        if self._last_move_increment_sec != cur_sec:
                            self.session_active_seconds += 1
                            self._last_move_increment_sec = cur_sec
            except Exception:
                pass
        def _format_elapsed_and_calories() -> str:
            if self.session_start_ts is None:
                return ""
            try:
                elapsed_sec = max(0, int(time.time() - self.session_start_ts))
                h = elapsed_sec // 3600
                m = (elapsed_sec % 3600) // 60
                s = elapsed_sec % 60
                # Calories only accrue while moving (active seconds)
                active_sec = max(0, int(self.session_active_seconds))
                duration_hours_active = active_sec / 3600.0
                kcal = self.activity_met * self.user_weight_kg * duration_hours_active
                if h > 0:
                    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    elapsed_str = f"{m:02d}:{s:02d}"
                return f" | Elapsed {elapsed_str} | {kcal:.1f} kcal"
            except Exception:
                return ""
        def _set():
            # Update active seconds based on MOVE flag if present
            _parse_move_and_maybe_increment()
            # Append session info if a session is active
            extras = _format_elapsed_and_calories()
            self.status_var.set(text + extras)
        try:
            self.after(0, _set)
        except Exception:
            pass

    def get_active_seconds(self) -> int:
        return int(self.session_active_seconds or 0)

    def on_new_frame(self, b64_png_bytes: bytes):
        # Update image on main thread
        def _update():
            try:
                img = tk.PhotoImage(data=b64_png_bytes)
                self.tk_image = img
                self.video_label.configure(image=img)
            except tk.TclError:
                # Some Tk builds may not support PNG data; show a message once
                self.set_status("Error: Your Tkinter build does not support PNG images.")
        self.after(0, _update)

    def refresh_cameras(self):
        if cv2 is None:
            details = f"\n\nDetails: {CV2_IMPORT_ERROR}" if CV2_IMPORT_ERROR else ""
            tip = ("\n\nTry: pip install opencv-python\n"
                   "If you are on a server or have display issues, try: pip install opencv-python-headless\n"
                   "On Linux you may also need: sudo apt-get install -y libgl1")
            messagebox.showerror("Missing Dependency", "OpenCV (cv2) is not available." + details + "\n" + tip)
            return
        self.set_status("Scanning cameras...")
        self.update_idletasks()
        enumerator = CameraEnumerator()
        cams = enumerator.list_cameras()
        display_items = [f"Camera {i}" for i in cams]
        self.cam_combo['values'] = display_items
        if cams:
            self.cam_combo.current(0)
            self.set_status(f"Found {len(cams)} camera(s). Select one and press Start.")
        else:
            self.set_status("No cameras found. Try Refresh or check permissions.")

    def get_selected_camera_index(self) -> Optional[int]:
        sel = self.cam_combo.get()
        if not sel:
            return None
        try:
            return int(sel.split()[-1])
        except Exception:
            return None

    def start_stream(self):
        if self.processor is not None:
            return
        cam_idx = self.get_selected_camera_index()
        if cam_idx is None:
            messagebox.showwarning("Select Camera", "Please select a camera from the list.")
            return
        # If MediaPipe is not installed, check for YOLO Pose alternative; otherwise run as basic viewer
        if mp is None:
            if YOLO is not None:
                self.set_status("MediaPipe not installed: using Ultralytics YOLO Pose backend.")
            else:
                self.set_status("MediaPipe not installed: running basic webcam viewer (no pose overlays).")
        if cv2 is None:
            details = f"\n\nDetails: {CV2_IMPORT_ERROR}" if CV2_IMPORT_ERROR else ""
            tip = ("\n\nTry: pip install opencv-python\n"
                   "If you are on a server or have display issues, try: pip install opencv-python-headless\n"
                   "On Linux you may also need: sudo apt-get install -y libgl1")
            messagebox.showerror("Missing Dependency", "OpenCV (cv2) is not available." + details + "\n" + tip)
            return

        # Start a new session timer for calorie estimation
        self.session_start_ts = time.time()
        self.session_active_seconds = 0
        self._last_move_increment_sec = None

        self.stop_event.clear()
        self.processor = VideoProcessor(
            cam_index=cam_idx,
            frame_callback=self.on_new_frame,
            status_callback=self.set_status,
            stop_event=self.stop_event,
            weight_kg=self.user_weight_kg,
            activity_met=self.activity_met,
            overlay_calories=self.overlay_calories,
            active_seconds_provider=self.get_active_seconds,
        )
        self.processor.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.refresh_btn.config(state=tk.DISABLED)
        self.cam_combo.config(state="disabled")
        # Delay enabling Record until camera is ready to avoid race conditions
        self.record_btn.config(state=tk.DISABLED)
        self.after(100, self._wait_until_ready)
        self.set_status(f"Starting camera {cam_idx}...")

    def _wait_until_ready(self, retries: int = 0):
        try:
            if self.processor and self.processor.cap:
                cap = self.processor.cap
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    if w > 0 and h > 0:
                        self.record_btn.config(state=tk.NORMAL)
                        return
        except Exception:
            pass
        # Retry for up to ~5 seconds
        if retries < 50:
            try:
                self.after(100, lambda: self._wait_until_ready(retries + 1))
            except Exception:
                pass
        else:
            # As a fallback, enable record after waiting in case some backends don't report size
            self.record_btn.config(state=tk.NORMAL)

    def stop_stream(self):
        if self.processor is None:
            return
        # Ensure recording stops
        if self.is_recording:
            try:
                self._stop_recording()
            except Exception:
                pass
        # Compute final session stats before clearing
        final_msg = None
        if self.session_start_ts is not None:
            try:
                elapsed_sec = max(0, int(time.time() - self.session_start_ts))
                active_sec = max(0, int(self.session_active_seconds))
                duration_hours_active = active_sec / 3600.0
                kcal = self.activity_met * self.user_weight_kg * duration_hours_active
                h = elapsed_sec // 3600
                m = (elapsed_sec % 3600) // 60
                s = elapsed_sec % 60
                if h > 0:
                    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
                else:
                    elapsed_str = f"{m:02d}:{s:02d}"
                ah = active_sec // 3600
                am = (active_sec % 3600) // 60
                a_s = active_sec % 60
                if ah > 0:
                    active_str = f"{ah:02d}:{am:02d}:{a_s:02d}"
                else:
                    active_str = f"{am:02d}:{a_s:02d}"
                final_msg = f"Session: Elapsed {elapsed_str}, Active {active_str}, {kcal:.1f} kcal"
            except Exception:
                final_msg = None
        self.set_status("Stopping...")
        self.stop_event.set()
        # Wait a short while for the thread to finish
        self.processor.join(timeout=2.0)
        self.processor = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.refresh_btn.config(state=tk.NORMAL)
        self.cam_combo.config(state="readonly")
        self.record_btn.config(state=tk.DISABLED)
        # Clear session timer after computing
        self.session_start_ts = None
        if final_msg:
            try:
                messagebox.showinfo("Session Summary", final_msg)
            except Exception:
                pass
            self.set_status(f"Stopped. {final_msg}")
        else:
            self.set_status("Stopped.")

    def on_close(self):
        try:
            self.stop_stream()
        except Exception:
            pass
        self.destroy()

    def toggle_record(self):
        if self.processor is None:
            try:
                messagebox.showwarning("Start Stream", "Please start the camera stream before recording.")
            except Exception:
                pass
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        # Ensure target directory exists and is writable; if not, fall back to a home-based folder
        def is_writable(dir_path: str) -> bool:
            try:
                os.makedirs(dir_path, exist_ok=True)
                test_path = os.path.join(dir_path, ".bt_write_test")
                with open(test_path, 'wb') as f:
                    f.write(b'ok')
                os.remove(test_path)
                return True
            except Exception:
                return False

        target_dir = self.records_dir
        if not is_writable(target_dir):
            home = os.path.expanduser("~")
            candidates = [
                os.path.join(home, "Videos", "body-tracker", "saved-tracks"),
                os.path.join(home, "Documents", "body-tracker", "saved-tracks"),
                os.path.join(home, ".body-tracker", "saved-tracks"),
            ]
            for cand in candidates:
                if is_writable(cand):
                    target_dir = cand
                    self.records_dir = cand
                    try:
                        messagebox.showinfo(
                            "Recording Folder",
                            f"The default folder was not writable. Using this folder instead:\n{os.path.abspath(cand)}"
                        )
                    except Exception:
                        pass
                    break
        # Proceed to create file path
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(target_dir, f"track-{ts}.mp4")
        ok = False
        try:
            ok = self.processor.start_recording(path)
        except Exception:
            ok = False
        if ok:
            self.is_recording = True
            self.record_btn.config(text="Stop Rec")
            try:
                abs_path = os.path.abspath(path)
            except Exception:
                abs_path = path
            self.current_record_path = abs_path
            try:
                self.set_status(f"Recording... {abs_path}")
            except Exception:
                self.set_status(f"Recording... {os.path.basename(path)}")
            try:
                messagebox.showinfo("Recording Started", f"Recording to:\n{abs_path}")
            except Exception:
                pass
        else:
            # Fallback attempt with .avi
            avi_path = os.path.join(target_dir, f"track-{ts}.avi")
            try:
                ok2 = self.processor.start_recording(avi_path)
            except Exception:
                ok2 = False
            if ok2:
                self.is_recording = True
                self.record_btn.config(text="Stop Rec")
                try:
                    abs_avi = os.path.abspath(avi_path)
                except Exception:
                    abs_avi = avi_path
                self.current_record_path = abs_avi
                try:
                    self.set_status(f"Recording... {abs_avi}")
                except Exception:
                    self.set_status(f"Recording... {os.path.basename(avi_path)}")
                try:
                    messagebox.showinfo("Recording Started", f"Recording to:\n{abs_avi}")
                except Exception:
                    pass
            else:
                messagebox.showerror("Recording Error", "Failed to start recording. Check write permissions and codecs.")

    def _stop_recording(self):
        try:
            self.processor.stop_recording()
        except Exception:
            pass
        # Inform user about saved file if we know the path
        try:
            if self.current_record_path:
                messagebox.showinfo("Recording Stopped", f"Recording saved to:\n{self.current_record_path}")
        except Exception:
            pass
        self.is_recording = False
        self.record_btn.config(text="Record")
        self.current_record_path = None

    def show_settings(self):
        win = tk.Toplevel(self)
        win.title("Settings")
        win.transient(self)
        try:
            win.grab_set()
        except Exception:
            pass
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        # Weight
        ttk.Label(frm, text="Weight (kg):").grid(row=0, column=0, sticky="w", pady=4)
        weight_var = tk.StringVar(value=str(self.user_weight_kg))
        ttk.Entry(frm, textvariable=weight_var, width=10).grid(row=0, column=1, sticky="w", pady=4)
        # MET
        ttk.Label(frm, text="Activity MET:").grid(row=1, column=0, sticky="w", pady=4)
        met_var = tk.StringVar(value=str(self.activity_met))
        ttk.Entry(frm, textvariable=met_var, width=10).grid(row=1, column=1, sticky="w", pady=4)
        # Overlay calories
        overlay_var = tk.BooleanVar(value=self.overlay_calories)
        ttk.Checkbutton(frm, text="Overlay calories on video", variable=overlay_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8,4))
        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=(8,0))
        def on_save():
            try:
                self.user_weight_kg = float(weight_var.get())
            except Exception:
                try:
                    messagebox.showwarning("Invalid Value", "Weight must be a number.")
                except Exception:
                    pass
                return
            try:
                self.activity_met = float(met_var.get())
            except Exception:
                try:
                    messagebox.showwarning("Invalid Value", "MET must be a number.")
                except Exception:
                    pass
                return
            self.overlay_calories = bool(overlay_var.get())
            try:
                messagebox.showinfo("Settings Saved", "Settings will apply to new streams. Stop/Start the stream to apply now.")
            except Exception:
                pass
            win.destroy()
        ttk.Button(btns, text="Save", command=on_save).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=6)

    def show_help(self):
        help_text = (
            "Body Tracker - Webcam Pose Scanner\n\n"
            "Instructions:\n"
            "1. Click Refresh to scan for available webcams.\n"
            "2. Choose a camera from the dropdown.\n"
            "3. Click Start to begin scanning body movements.\n"
            "4. (Optional) Click Record to start/stop saving the video into the 'saved-tracks' folder.\n"
            "5. Click Stop to end the stream.\n\n"
            "Tip: Use the Settings menu to adjust Weight (kg), Activity MET, and toggle the calories overlay.\n\n"
            "Dependencies:\n"
            "  Required: opencv-python\n"
            "  Optional: mediapipe (for holistic pose/face/hands), ultralytics (for YOLOv8/YOLOv10 Pose)\n\n"
            "Notes:\n"
            "- With MediaPipe installed, the app draws pose, face, and hands landmarks (head/eyes/mouth and body with hands/feet).\n"
            "- If MediaPipe is not installed but Ultralytics is, the app uses YOLO Pose to draw the body skeleton.\n"
            "- Without both, the app runs as a basic webcam viewer (no overlays).\n"
            "- If no cameras are found, ensure permissions are granted and no other app is using the camera.\n"
            "- FPS, elapsed time, and calories appear at the bottom during a session.\n"
            "- Elapsed time and calories overlay are drawn on the video and saved into recordings (if enabled in Settings).\n\n"
            "Calories:\n"
            "- Calories increase only when movement is detected. We estimate: MET * weight_kg * (active_seconds/3600).\n"
            "- Configure via environment variables before starting:\n"
            "    BT_WEIGHT_KG (default 70), BT_ACTIVITY_MET (default 3.0 for light activity).\n"
            "- Motion detection thresholds (advanced): BT_MOTION_DIFF_THRESH (15), BT_MOTION_RATIO_THRESH (0.005), BT_MOTION_FRAMES_RATIO (0.1).\n"
            "- Movement sensitivity mode: BT_MOTION_MODE=full|any (default: full). 'full' requires both upper (shoulders) and lower body (hips) to move; 'any' keeps the previous behavior.\n"
            "- Full-body pixel threshold: BT_BODY_PIX_THRESH (default: 8). Increase to make MOVE harder to trigger; decrease to be more sensitive.\n"
        )
        messagebox.showinfo("Help", help_text)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
