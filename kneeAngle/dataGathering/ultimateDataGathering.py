import cv2
import numpy as np
import csv
import threading
import serial
import time
from collections import deque
from datetime import datetime
from scipy.signal import butter, filtfilt
import os
import shutil

# ================================================================
# SETTINGS
# ================================================================
INPUT_CAM  = 2
VTS_DEVICE = 0
RECORD_FPS = 30.0

# ---- Dynamometer ----
DYNO_PORT          = "COM6"
DYNO_BAUD          = 115200
DYNO_GRAVITY       = 9.80665
DYNO_TARE_SAMPLES  = 150
DYNO_NOISE_THRESH  = 0.05        # N — readings below this → 0
DYNO_BUFFER_SIZE   = 5000        # rolling buffer of (wall_time, force_N, mass_kg)

# All sessions share one timestamp so folders are always paired
_TS = datetime.now().strftime('%Y%m%d_%H%M%S')

# ---- Output roots ----
_OUT_ROOT  = r".\kneeAngle\updated\output"
_KNEE_ROOT = os.path.join(_OUT_ROOT, "knee")
_VTS_ROOT  = os.path.join(_OUT_ROOT, "vts")
_CSV_ROOT  = os.path.join(_OUT_ROOT, "outputCSV")
_TEMP_ROOT = os.path.join(_OUT_ROOT, "_temp")

# ---- Per-session folders ----
_KNEE_FOLDER = os.path.join(_KNEE_ROOT, f"knee_{_TS}")
_VTS_FOLDER  = os.path.join(_VTS_ROOT,  f"vts_{_TS}")

# ---- Output paths ----
OUTPUT_KNEE_RAW        = os.path.join(_KNEE_FOLDER, "knee_raw.mp4")
OUTPUT_KNEE_ANNOTATED  = os.path.join(_KNEE_FOLDER, "knee_annotated.mp4")
OUTPUT_VTS_RAW         = os.path.join(_VTS_FOLDER,  "vts_raw.mp4")
OUTPUT_VTS_PROCESSED   = os.path.join(_VTS_FOLDER,  "processedVer.mp4")
OUTPUT_CSV             = os.path.join(_CSV_ROOT,     f"knee_angle_tracked_{_TS}.csv")

# ---- Frame dump sub-dirs ----
_KNEE_FRAMES_RAW       = os.path.join(_KNEE_FOLDER, "frames", "raw")
_KNEE_FRAMES_ANNOTATED = os.path.join(_KNEE_FOLDER, "frames", "annotated")
_VTS_FRAMES_PROCESSED  = os.path.join(_VTS_FOLDER,  "frames")

# ---- Temp files (overwritten each session) ----
_TEMP_KNEE = os.path.join(_TEMP_ROOT, "_temp_knee.mp4")
_TEMP_VTS  = os.path.join(_TEMP_ROOT, "_temp_vts.mp4")

# ---- LK / tracking ----
SUBJECT_ID = "S1"
SIDE       = "LEFT"

LK_PARAMS = dict(
    winSize  = (41, 41),
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
MIN_TRACK_CONFIDENCE = 0.3
FB_ERROR_THRESHOLD   = 3.0

JOINT_NAMES  = ["HIP", "KNEE", "ANKLE"]
JOINT_COLORS = [(0, 200, 255), (0, 255, 100), (80, 80, 255)]

# ---- HUD palette ----
HUD_BG     = (15,  15,  15)
HUD_ACCENT = (0,  210, 255)
HUD_DIM    = (120, 120, 120)
HUD_WHITE  = (230, 230, 230)
HUD_GREEN  = (60,  220, 100)
HUD_RED    = (60,   60, 220)
HUD_YELLOW = (0,  210, 255)
REC_RED    = (30,   30, 220)
HUD_ORANGE = (30,  140, 255)

DISPLAY_EVERY_N_FRAMES = 3

# ---- Eulerian motion magnification settings (VTS only) ----
EMM_SCALE_FACTOR   = 0.5
EMM_AMPLIFICATION  = 30
EMM_FREQ_MIN       = 0.8
EMM_FREQ_MAX       = 2.0
EMM_PYRAMID_LEVELS = 4
EMM_K1             = -0.05
EMM_FX_SCALE       = 0.9
EMM_FY_SCALE       = 0.9
EMM_CROP = {
    "LEFT":   0.23,
    "RIGHT":  0.20,
    "TOP":    0.20,
    "BOTTOM": 0.17,
}


# ================================================================
# DYNAMOMETER  — serial reader (runs in background from startup)
# ================================================================

class Dynamometer:
    """
    Reads force data from a load-cell over serial in a background thread.
    Stores a rolling deque of (wall_time, force_N, mass_kg) tuples so that
    any frame can be matched by wall-clock timestamp during CSV export.

    Key bindings (handled in main loop):
        F  — set rep-zero (replaces SPACE, which is used for record/stop)
        T  — re-tare (global)
    """

    def __init__(self, port, baud, buffer_size=DYNO_BUFFER_SIZE):
        self.port        = port
        self.baud        = baud
        self._lock       = threading.Lock()
        self._running    = False
        self._thread     = None

        # tare & rep-zero state
        self.tare_offset     = 0.0
        self.rep_zero_offset = 0.0

        # latest live values (for HUD)
        self._latest_raw    = 0.0
        self._latest_force  = 0.0
        self._latest_mass   = 0.0

        # timestamped history for frame alignment
        self.history = deque(maxlen=buffer_size)   # (wall_time, force_N, mass_kg)

        self.ser    = None
        self.ready  = False          # True once tare is done
        self.status = "NOT CONNECTED"

    # ----------------------------------------------------------------
    def connect_and_tare(self):
        """Open serial port and perform initial global tare. Blocking."""
        try:
            print(f"[DYNO] Opening {self.port} @ {self.baud}...")
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            self._do_tare(label="startup")
            self.ready  = True
            self.status = "READY"
            print(f"[DYNO] Ready. Tare offset = {self.tare_offset:.3f} g")
        except Exception as e:
            print(f"[DYNO] WARNING — could not connect: {e}")
            print("[DYNO] Continuing without dynamometer. Force values will be 0.")
            self.status = "DISCONNECTED"
            self.ready  = True   # allow pipeline to proceed without force

    def _do_tare(self, label=""):
        """Read DYNO_TARE_SAMPLES raw values and update tare_offset. Blocking."""
        print(f"[DYNO] Taring ({label}) — keep device unloaded...")
        samples = []
        while len(samples) < DYNO_TARE_SAMPLES:
            line = self.ser.readline().decode(errors="ignore").strip()
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    samples.append(float(parts[1]))
                except ValueError:
                    pass
        self.tare_offset = sum(samples) / len(samples)
        print(f"[DYNO] Tare done: offset = {self.tare_offset:.3f} g")

    # ----------------------------------------------------------------
    def retare(self):
        """Triggered by 'T' key in the main loop."""
        if self.ser and self.ser.isOpen():
            self._do_tare(label="manual retare")

    def set_rep_zero(self):
        """Triggered by 'F' key — zeroes the current resting position."""
        with self._lock:
            self.rep_zero_offset = self._latest_raw - self.tare_offset
        print(f"[DYNO] Rep-zero set. Offset = {self.rep_zero_offset:.3f} g")

    # ----------------------------------------------------------------
    def start(self):
        """Launch background serial reading thread."""
        if self.ser is None:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.ser and self.ser.isOpen():
            self.ser.close()

    # ----------------------------------------------------------------
    def _read_loop(self):
        while self._running:
            try:
                line  = self.ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                raw_mass = float(parts[1])
                t        = time.time()

                with self._lock:
                    self._latest_raw = raw_mass
                    corrected = raw_mass - self.tare_offset - self.rep_zero_offset
                    mass_kg   = corrected / 1000.0
                    force_N   = mass_kg * DYNO_GRAVITY
                    if force_N < DYNO_NOISE_THRESH:
                        force_N = 0.0
                        mass_kg = 0.0
                    self._latest_force = force_N
                    self._latest_mass  = mass_kg

                self.history.append((t, force_N, mass_kg))

            except Exception:
                pass

    # ----------------------------------------------------------------
    @property
    def live_force(self):
        with self._lock:
            return self._latest_force

    @property
    def live_mass(self):
        with self._lock:
            return self._latest_mass

    def get_sample_at(self, wall_time):
        """Return (force_N, mass_kg) closest in time to wall_time."""
        with self._lock:
            hist = list(self.history)
        if not hist:
            return 0.0, 0.0
        # binary-search nearest timestamp
        times = [s[0] for s in hist]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - wall_time))
        return hist[idx][1], hist[idx][2]


# ---- Initialise dynamometer (blocks for tare) ----
dyno = Dynamometer(DYNO_PORT, DYNO_BAUD)
dyno.connect_and_tare()
dyno.start()


# ================================================================
# Utilities
# ================================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc  = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def draw_skeleton(frame, pts, confidences, angle):
    all_ok     = all(c > MIN_TRACK_CONFIDENCE for c in confidences)
    line_color = (0, 230, 120) if all_ok else (0, 165, 255)
    if len(pts) >= 2:
        cv2.line(frame, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), line_color, 3)
    if len(pts) == 3:
        cv2.line(frame, tuple(pts[1].astype(int)), tuple(pts[2].astype(int)), line_color, 3)
    for i, (pt, conf) in enumerate(zip(pts, confidences)):
        p  = tuple(pt.astype(int))
        ok = conf > MIN_TRACK_CONFIDENCE
        cv2.circle(frame, p, 9, JOINT_COLORS[i] if ok else (60, 60, 200), -1)
        cv2.circle(frame, p, 9, (255, 255, 255), 2)
        cv2.putText(frame, JOINT_NAMES[i], (p[0]+12, p[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, JOINT_COLORS[i], 2)
    if angle is not None and len(pts) == 3:
        knee = tuple(pts[1].astype(int))
        cv2.putText(frame, f"{angle:.1f} deg", (knee[0]+20, knee[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
    return frame


def draw_force_overlay(frame, force_N, mass_kg):
    """Draw a small force readout in the top-left corner."""
    h, w    = frame.shape[:2]
    pad     = 10
    bar_h   = 60
    roi     = frame[0:bar_h, 0:220]
    blended = cv2.addWeighted(roi, 0.25, np.full_like(roi, HUD_BG), 0.75, 0)
    frame[0:bar_h, 0:220] = blended
    col = HUD_GREEN if force_N > DYNO_NOISE_THRESH else HUD_DIM
    cv2.putText(frame, f"F: {force_N:6.1f} N", (pad, pad+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"M: {mass_kg*1000:6.0f} g", (pad, pad+42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_DIM, 1, cv2.LINE_AA)
    if dyno.status == "DISCONNECTED":
        cv2.putText(frame, "NO DYNO", (pad+120, pad+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUD_RED, 2, cv2.LINE_AA)
    return frame


# ----------------------------------------------------------------
# Generic HUD helpers
# ----------------------------------------------------------------

def draw_hud_bar(frame, lines, position="bottom", alpha=0.72):
    h, w    = frame.shape[:2]
    padding = 12
    line_h  = 32
    bar_h   = padding * 2 + line_h * len(lines)
    y0 = (h - bar_h) if position == "bottom" else 0
    y1 = h            if position == "bottom" else bar_h
    roi     = frame[y0:y1, 0:w]
    blended = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, HUD_BG), alpha, 0)
    frame[y0:y1, 0:w] = blended
    accent_y = y0 if position == "bottom" else y1
    cv2.line(frame, (0, accent_y), (w, accent_y), HUD_ACCENT, 2)
    for i, (text, color, scale, thick) in enumerate(lines):
        ty = y0 + padding + line_h * i + 20
        cv2.putText(frame, text, (padding+6, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return frame


def draw_status_badge(frame, text, color, position="top-right"):
    h, w = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 10
    bw, bh = tw + pad*2, th + pad*2
    if   position == "top-right": x0, y0 = w - bw - 10, 10
    elif position == "top-left":  x0, y0 = 10, 10
    else:                         x0, y0 = 10, 10
    roi     = frame[y0:y0+bh, x0:x0+bw]
    blended = cv2.addWeighted(roi, 0.15, np.full_like(roi, color), 0.85, 0)
    frame[y0:y0+bh, x0:x0+bw] = blended
    cv2.rectangle(frame, (x0, y0), (x0+bw, y0+bh), color, 2)
    cv2.putText(frame, text, (x0+pad, y0+bh-pad),
                font, scale, HUD_WHITE, thick, cv2.LINE_AA)
    return frame


# ----------------------------------------------------------------
# Click-joints UI
# ----------------------------------------------------------------

_vignette_cache: dict = {}

def _get_vignette(h, w):
    key = (h, w)
    if key not in _vignette_cache:
        vig = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(vig, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
        vig  = cv2.GaussianBlur(vig, (0,0), w//4)
        mask = cv2.cvtColor(vig, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        _vignette_cache[key] = mask
    return _vignette_cache[key]


# ----------------------------------------------------------------
# Recording / review HUD overlays
# ----------------------------------------------------------------

def draw_preview_hud(frame, state):
    draw_hud_bar(frame, [
        ("LIVE PREVIEW", HUD_ACCENT, 0.72, 2),
        ("SPACE = start recording   Q = quit   F = rep-zero dyno   T = retare dyno", HUD_DIM, 0.45, 1),
    ], position="bottom")
    draw_status_badge(frame, "STANDBY", HUD_DIM, "top-right")
    draw_force_overlay(frame, dyno.live_force, dyno.live_mass)
    return frame


def draw_recording_hud(frame):
    draw_hud_bar(frame, [
        ("SPACE = stop recording   Q = quit   F = rep-zero dyno   T = retare dyno", HUD_DIM, 0.45, 1),
    ], position="bottom")
    draw_status_badge(frame, "● REC", REC_RED, "top-right")
    draw_force_overlay(frame, dyno.live_force, dyno.live_mass)
    return frame


def draw_review_hud(frame):
    draw_hud_bar(frame, [
        ("REVIEWING RECORDING", HUD_YELLOW, 0.72, 2),
        ("ENTER = proceed   R = re-record   SPACE = replay   Q = quit", HUD_DIM, 0.50, 1),
    ], position="bottom")
    draw_status_badge(frame, "REVIEW", (30, 160, 210), "top-right")
    return frame


def draw_vts_hud(frame, state, fi=None, total=None, angle=None):
    if state == "PREVIEW":
        draw_hud_bar(frame, [("VTS | LIVE", HUD_ACCENT, 0.65, 2)], position="bottom")
        draw_status_badge(frame, "STANDBY", HUD_DIM, "top-right")
    elif state == "RECORDING":
        draw_hud_bar(frame, [("VTS | RECORDING", REC_RED, 0.65, 2)], position="bottom")
        draw_status_badge(frame, "● REC", REC_RED, "top-right")
    elif state == "REVIEW":
        draw_hud_bar(frame, [("VTS | REVIEW", HUD_YELLOW, 0.65, 2)], position="bottom")
    elif state == "TRACKING":
        lines = [("VTS | TRACKING", HUD_GREEN, 0.65, 2)]
        if fi is not None and total is not None:
            lines.append((f"Frame {fi+1} / {total}", HUD_DIM, 0.50, 1))
        if angle is not None:
            lines.append((f"Angle: {angle:.1f} deg", HUD_ACCENT, 0.60, 2))
        draw_hud_bar(frame, lines, position="bottom")
    elif state == "WRITING":
        lines = [("VTS | WRITING OUTPUT", HUD_ACCENT, 0.65, 2)]
        if fi is not None and total is not None:
            lines.append((f"Frame {fi+1} / {total}", HUD_DIM, 0.50, 1))
        draw_hud_bar(frame, lines, position="bottom")
    elif state == "CLICKING":
        draw_hud_bar(frame, [
            ("VTS | JOINT SELECTION", HUD_ACCENT, 0.65, 2),
            ("Place HIP → KNEE → ANKLE on Main Cam", HUD_DIM, 0.50, 1),
        ], position="bottom")
    elif state == "EMM":
        lines = [("VTS | EULERIAN PROCESSING (background)", HUD_GREEN, 0.60, 2)]
        if fi is not None and total is not None:
            lines.append((f"Step {fi} / {total}", HUD_DIM, 0.50, 1))
        draw_hud_bar(frame, lines, position="bottom")
    draw_force_overlay(frame, dyno.live_force, dyno.live_mass)
    return frame


# ================================================================
# Camera / writer helpers
# ================================================================

def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None, 0, 0, RECORD_FPS
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or RECORD_FPS
    return cap, w, h, fps


def make_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))


def init_windows(kw, kh, vw, vh):
    cv2.namedWindow("Main Cam",    cv2.WINDOW_NORMAL)
    cv2.namedWindow("VTS Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main Cam",    kw, kh)
    cv2.resizeWindow("VTS Preview", vw, vh)


# ================================================================
# Frame-dump helper
# ================================================================

def dump_frames(video_path: str, frames_dir: str, prefix: str = "frame") -> int:
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[dump_frames] Cannot open {video_path}")
        return 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{prefix}_{idx:06d}.png"), frame)
        idx += 1
    cap.release()
    print(f"[dump_frames] Saved {idx} frames → {frames_dir}")
    return idx


# ================================================================
# Eulerian Motion Magnification  (runs in background thread)
# ================================================================

def _build_laplacian_pyramid(img, levels):
    pyramid, temp = [], img.copy()
    for _ in range(levels):
        down = cv2.pyrDown(temp)
        up   = cv2.pyrUp(down, dstsize=(temp.shape[1], temp.shape[0]))
        pyramid.append(cv2.subtract(temp, up))
        temp = down
    pyramid.append(temp)
    return pyramid


def _collapse_laplacian_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        up  = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = cv2.add(up, pyramid[i])
    return img


def _temporal_bandpass(data, fps, fmin, fmax):
    nyquist = 0.5 * fps
    low  = max(fmin / nyquist, 1e-4)
    high = min(fmax / nyquist, 1 - 1e-4)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def _fisheye_correct(frame, K, D):
    return cv2.undistort(frame, K, D)


def eulerian_process_vts(input_path: str, output_path: str, frames_dir: str):
    print("[EMM] Starting Eulerian processing...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[EMM] Cannot open {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or RECORD_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fx = EMM_FX_SCALE * width
    fy = EMM_FY_SCALE * height
    K  = np.array([[fx, 0, width/2],
                   [0, fy, height/2],
                   [0,  0, 1       ]], dtype=np.float32)
    D  = np.array([EMM_K1, 0, 0, 0], dtype=np.float32)

    crop_l = int(EMM_CROP["LEFT"]   * width)
    crop_r = int(EMM_CROP["RIGHT"]  * width)
    crop_t = int(EMM_CROP["TOP"]    * height)
    crop_b = int(EMM_CROP["BOTTOM"] * height)

    frames_yuv = []
    scaled_w = scaled_h = None

    print(f"[EMM] Loading {total} frames...")
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        corrected = _fisheye_correct(frame, K, D)
        cropped   = corrected[crop_t: height - crop_b, crop_l: width - crop_r]
        cropped   = cv2.resize(
            cropped,
            (int(cropped.shape[1] * EMM_SCALE_FACTOR),
             int(cropped.shape[0] * EMM_SCALE_FACTOR)),
            interpolation=cv2.INTER_AREA
        )
        if scaled_w is None:
            scaled_h, scaled_w = cropped.shape[:2]
        yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV).astype(np.float32) / 255.0
        frames_yuv.append(yuv)
        if i % 100 == 0:
            print(f"[EMM]   loaded {i}/{total}")
    cap.release()

    N = len(frames_yuv)
    if N == 0:
        print("[EMM] No frames loaded — aborting.")
        return

    print("[EMM] Building pyramids...")
    pyramid_seq = []
    for i, frame in enumerate(frames_yuv):
        pyramid_seq.append(_build_laplacian_pyramid(frame[:, :, 0], EMM_PYRAMID_LEVELS))
        if i % 100 == 0:
            print(f"[EMM]   pyramid {i}/{N}")

    num_levels = len(pyramid_seq[0])

    print("[EMM] Temporal filtering...")
    filtered_pyramids = []
    for lv in range(num_levels):
        seq  = np.array([pyramid_seq[t][lv] for t in range(N)], dtype=np.float32)
        filt = _temporal_bandpass(seq, fps, EMM_FREQ_MIN, EMM_FREQ_MAX)
        filtered_pyramids.append(filt)
        print(f"[EMM]   filtered level {lv+1}/{num_levels}")

    print("[EMM] Reconstructing...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = make_writer(output_path, fps, scaled_w, scaled_h)
    os.makedirs(frames_dir, exist_ok=True)

    for t in range(N):
        orig_pyr = pyramid_seq[t]
        mag_pyr  = [
            orig_pyr[lv] + EMM_AMPLIFICATION * filtered_pyramids[lv][t]
            for lv in range(num_levels)
        ]
        y_mag   = np.clip(_collapse_laplacian_pyramid(mag_pyr), 0, 1)
        yuv_mag = np.stack([y_mag,
                            frames_yuv[t][:, :, 1],
                            frames_yuv[t][:, :, 2]], axis=2)
        out = cv2.cvtColor((yuv_mag * 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
        writer.write(out)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{t:06d}.png"), out)
        if t % 100 == 0:
            print(f"[EMM]   writing {t}/{N}")

    writer.release()
    print(f"[EMM] Done → {output_path}")
    print(f"[EMM] Frames → {frames_dir}")


# ================================================================
# Phase 1 — Preview → Record → Review
# ================================================================

def record_phase():
    """
    Returns (kfps, kw, kh, vfps, vw, vh, rec_start_time, rec_end_time)
    rec_start_time / rec_end_time are wall-clock times so dynamometer
    samples can be trimmed and aligned to video frames.
    """
    knee_cap, kw, kh, kfps = open_camera(INPUT_CAM)
    vts_cap,  vw, vh, vfps = open_camera(VTS_DEVICE)
    if knee_cap is None:
        raise RuntimeError(f"Cannot open main camera (device {INPUT_CAM})")
    if vts_cap is None:
        raise RuntimeError(f"Cannot open VTS camera (device {VTS_DEVICE})")

    print(f"Main cam : {kw}x{kh} @ {kfps:.1f}fps")
    print(f"VTS cam  : {vw}x{vh} @ {vfps:.1f}fps")
    os.makedirs(os.path.abspath(_TEMP_ROOT), exist_ok=True)

    init_windows(kw, kh, vw, vh)

    knee_writer    = None
    vts_writer     = None
    state          = "PREVIEW"
    knee_pb        = None
    vts_pb         = None
    rec_start_time = None
    rec_end_time   = None

    print("PREVIEW — SPACE=record  F=rep-zero dyno  T=retare  Q=quit")

    while True:
        if state in ("PREVIEW", "RECORDING"):
            ret_k, kframe = knee_cap.read()
            ret_v, vframe = vts_cap.read()

            if state == "RECORDING":
                if ret_k and knee_writer: knee_writer.write(kframe)
                if ret_v and vts_writer:  vts_writer.write(vframe)

            if ret_k:
                kdisp = kframe.copy()
                draw_preview_hud(kdisp, state) if state == "PREVIEW" else draw_recording_hud(kdisp)
                cv2.imshow("Main Cam", kdisp)

            if ret_v:
                vdisp = vframe.copy()
                draw_vts_hud(vdisp, state)
                cv2.imshow("VTS Preview", vdisp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if state == "PREVIEW":
                    knee_writer    = make_writer(_TEMP_KNEE, kfps, kw, kh)
                    vts_writer     = make_writer(_TEMP_VTS,  vfps, vw, vh)
                    rec_start_time = time.time()
                    state          = "RECORDING"
                    print("Recording...")
                elif state == "RECORDING":
                    rec_end_time = time.time()
                    knee_writer.release(); knee_writer = None
                    vts_writer.release();  vts_writer  = None
                    knee_cap.release();    knee_cap    = None
                    vts_cap.release();     vts_cap     = None
                    state  = "REVIEW"
                    print(f"Stopped recording. Duration = {rec_end_time - rec_start_time:.2f}s")
                    knee_pb = cv2.VideoCapture(_TEMP_KNEE)
                    vts_pb  = cv2.VideoCapture(_TEMP_VTS)
                    knee_pb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    vts_pb.set(cv2.CAP_PROP_POS_FRAMES, 0)

            elif key == ord('f'):
                dyno.set_rep_zero()

            elif key == ord('t'):
                t_thread = threading.Thread(target=dyno.retare, daemon=True)
                t_thread.start()

            elif key in (ord('q'), 27):
                if knee_cap:    knee_cap.release()
                if vts_cap:     vts_cap.release()
                if knee_writer: knee_writer.release()
                if vts_writer:  vts_writer.release()
                dyno.stop()
                cv2.destroyAllWindows(); exit()

        elif state == "REVIEW":
            ret_k, kframe = knee_pb.read()
            ret_v, vframe = vts_pb.read()

            if ret_k:
                kdisp = kframe.copy()
                draw_review_hud(kdisp)
                cv2.imshow("Main Cam", kdisp)
            if ret_v:
                vdisp = vframe.copy()
                draw_vts_hud(vdisp, "REVIEW")
                cv2.imshow("VTS Preview", vdisp)

            key = cv2.waitKey(int(1000 / 30)) & 0xFF

            if key in (13, ord('\n')):
                knee_pb.release(); vts_pb.release()
                print("Proceeding to joint selection...")
                return kfps, kw, kh, vfps, vw, vh, rec_start_time, rec_end_time

            elif key == ord('r'):
                knee_pb.release(); vts_pb.release()
                knee_pb = None; vts_pb = None
                knee_cap, kw, kh, kfps = open_camera(INPUT_CAM)
                vts_cap,  vw, vh, vfps = open_camera(VTS_DEVICE)
                init_windows(kw, kh, vw, vh)
                state = "PREVIEW"
                print("Re-recording — SPACE to start.")

            elif key == ord(' '):
                knee_pb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                vts_pb.set(cv2.CAP_PROP_POS_FRAMES, 0)

            elif key in (ord('q'), 27):
                knee_pb.release(); vts_pb.release()
                dyno.stop()
                cv2.destroyAllWindows(); exit()

            if not ret_k and not ret_v:
                knee_pb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                vts_pb.set(cv2.CAP_PROP_POS_FRAMES, 0)


# ================================================================
# Phase 2 — Click joints
# ================================================================

def click_joints_phase(kw, kh):
    cap = cv2.VideoCapture(_TEMP_KNEE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, first_frame = cap.read()
    cap.release()

    first_gray  = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    seed_clicks = []
    mouse_pos   = [None]

    _get_vignette(kh, kw)
    _base_dark = (first_frame.astype(np.float32) * _get_vignette(kh, kw)).astype(np.uint8)

    vts_cap, vw, vh, _ = open_camera(VTS_DEVICE)
    init_windows(kw, kh, vw if vts_cap else 640, vh if vts_cap else 480)

    def mouse_cb(event, x, y, flags, param):
        mouse_pos[0] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(seed_clicks) < 3:
            seed_clicks.append((x, y))

    cv2.setMouseCallback("Main Cam", mouse_cb)
    print("Click HIP → KNEE → ANKLE, then ENTER. Z=undo R=reset Q=quit")

    while True:
        kdisp = _base_dark.copy()
        h, w  = kdisp.shape[:2]

        if mouse_pos[0] is not None:
            mx, my = mouse_pos[0]
            arm, gap, col = 18, 6, HUD_ACCENT
            cv2.line(kdisp, (mx-arm,my), (mx-gap,my), col, 1, cv2.LINE_AA)
            cv2.line(kdisp, (mx+gap,my), (mx+arm,my), col, 1, cv2.LINE_AA)
            cv2.line(kdisp, (mx,my-arm), (mx,my-gap), col, 1, cv2.LINE_AA)
            cv2.line(kdisp, (mx,my+gap), (mx,my+arm), col, 1, cv2.LINE_AA)
            cv2.circle(kdisp, (mx,my), 3, col, -1, cv2.LINE_AA)

        for i, pt in enumerate(seed_clicks):
            px, py = int(pt[0]), int(pt[1])
            col    = JOINT_COLORS[i]
            cv2.circle(kdisp, (px,py), 14, col, 2, cv2.LINE_AA)
            cv2.circle(kdisp, (px,py),  8, col, -1, cv2.LINE_AA)
            cv2.circle(kdisp, (px,py),  8, (255,255,255), 2, cv2.LINE_AA)
            if i < len(seed_clicks) - 1:
                nx, ny = int(seed_clicks[i+1][0]), int(seed_clicks[i+1][1])
                cv2.line(kdisp, (px,py), (nx,ny), col, 2, cv2.LINE_AA)
            lx, ly = px+18, py-10
            cv2.putText(kdisp, JOINT_NAMES[i], (lx+1,ly+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(kdisp, JOINT_NAMES[i], (lx,ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
            btext = f" {i+1} "
            cv2.putText(kdisp, btext, (px-6,py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(kdisp, btext, (px-6,py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_WHITE, 1, cv2.LINE_AA)

        placed = len(seed_clicks)
        if placed < 3:
            draw_hud_bar(kdisp, [
                (f"CLICK {JOINT_NAMES[placed]} ({placed+1} of 3)", JOINT_COLORS[placed], 0.80, 2),
            ], position="top")
        else:
            draw_hud_bar(kdisp, [
                ("ALL JOINTS PLACED — Press ENTER to confirm", HUD_GREEN, 0.80, 2),
            ], position="top")

        draw_hud_bar(kdisp, [
            ("LMB = place   Z = undo   R = reset   ENTER = confirm   Q = quit",
             HUD_DIM, 0.50, 1),
        ], position="bottom")

        dot_r, dot_gap = 7, 22
        sx = w - (3 * dot_gap) - 14
        for i in range(3):
            cx, cy = sx + i*dot_gap, 28
            if i < placed:
                cv2.circle(kdisp, (cx,cy), dot_r, JOINT_COLORS[i], -1,  cv2.LINE_AA)
                cv2.circle(kdisp, (cx,cy), dot_r, HUD_WHITE,        2,   cv2.LINE_AA)
            else:
                cv2.circle(kdisp, (cx,cy), dot_r, HUD_DIM,          1,   cv2.LINE_AA)

        cv2.imshow("Main Cam", kdisp)

        if vts_cap and vts_cap.isOpened():
            ret_v, vframe = vts_cap.read()
            if ret_v:
                vdisp = vframe.copy()
                draw_vts_hud(vdisp, "CLICKING")
                cv2.imshow("VTS Preview", vdisp)

        key = cv2.waitKey(20) & 0xFF
        if   key == ord('z') and seed_clicks: seed_clicks.pop()
        elif key == ord('r'):                 seed_clicks.clear()
        elif key in (13, ord('\n')) and len(seed_clicks) == 3:
            break
        elif key in (ord('q'), 27):
            if vts_cap: vts_cap.release()
            dyno.stop()
            cv2.destroyAllWindows(); exit()

    if vts_cap: vts_cap.release()
    return seed_clicks, first_gray


# ================================================================
# Phase 3 — Track + write all outputs
# ================================================================

def track_and_write_phase(seed_clicks, first_gray, kw, kh, kfps,
                          rec_start_time, rec_end_time):
    # ---- prepare output dirs ----
    for d in [_KNEE_FOLDER, _VTS_FOLDER,
              _KNEE_FRAMES_RAW, _KNEE_FRAMES_ANNOTATED,
              _VTS_FRAMES_PROCESSED,
              os.path.dirname(os.path.abspath(OUTPUT_CSV))]:
        os.makedirs(d, exist_ok=True)

    shutil.copy2(_TEMP_VTS, OUTPUT_VTS_RAW)
    print(f"VTS raw saved → {OUTPUT_VTS_RAW}")

    # ---- PASS 1: optical flow tracking ----
    cap   = cv2.VideoCapture(_TEMP_KNEE)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or kfps

    vts_cap, vw, vh, _ = open_camera(VTS_DEVICE)
    init_windows(kw, kh, vw if vts_cap else 640, vh if vts_cap else 480)

    prev_pts       = np.array(seed_clicks, dtype=np.float32).reshape(-1, 1, 2)
    last_good_pts  = prev_pts.reshape(-1, 2).copy()
    tracked_pts    = []
    tracked_angles = []
    tracked_lost   = []
    prev_gray      = first_gray.copy()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Tracking...")

    for fi in range(total):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lost = [False, False, False]

        if fi > 0:
            fwd_pts, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **LK_PARAMS)
            bwd_pts, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, fwd_pts, None, **LK_PARAMS)
            for i in range(3):
                both_ok = (st_fwd[i][0] == 1) and (st_bwd[i][0] == 1)
                if both_ok:
                    fb_err = np.linalg.norm(prev_pts[i][0] - bwd_pts[i][0])
                    if fb_err < FB_ERROR_THRESHOLD:
                        last_good_pts[i] = fwd_pts[i][0]
                    else:
                        lost[i] = True
                else:
                    lost[i] = True
            prev_pts = last_good_pts.reshape(-1,1,2).astype(np.float32)

        angle = calculate_angle(last_good_pts[0], last_good_pts[1], last_good_pts[2])
        tracked_pts.append(last_good_pts.copy())
        tracked_angles.append(angle)
        tracked_lost.append(lost)
        prev_gray = gray.copy()

        if fi % DISPLAY_EVERY_N_FRAMES == 0:
            kdisp = frame.copy()
            confs = [0.0 if lost[i] else 1.0 for i in range(3)]
            draw_skeleton(kdisp, last_good_pts, confs, angle)
            draw_hud_bar(kdisp, [(f"TRACKING {fi+1} / {total}", HUD_ACCENT, 0.65, 2)],
                         position="bottom")
            draw_force_overlay(kdisp, dyno.live_force, dyno.live_mass)
            cv2.imshow("Main Cam", kdisp)
            if vts_cap and vts_cap.isOpened():
                ret_v, vframe = vts_cap.read()
                if ret_v:
                    vdisp = vframe.copy()
                    draw_vts_hud(vdisp, "TRACKING", fi=fi, total=total, angle=angle)
                    cv2.imshow("VTS Preview", vdisp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                dyno.set_rep_zero()
            elif key == ord('t'):
                threading.Thread(target=dyno.retare, daemon=True).start()

    cap.release()
    if vts_cap: vts_cap.release()

    # ---- PASS 2: write knee_raw, knee_annotated, CSV (with force), dump frames ----
    raw_writer  = make_writer(OUTPUT_KNEE_RAW,       fps, kw, kh)
    ann_writer  = make_writer(OUTPUT_KNEE_ANNOTATED, fps, kw, kh)
    csv_file    = open(OUTPUT_CSV, 'w', newline='')
    csv_wr      = csv.writer(csv_file)
    csv_wr.writerow([
        "frame", "timestamp_s", "wall_time",
        "angle_deg",
        "hip_x",  "hip_y",
        "knee_x", "knee_y",
        "ankle_x","ankle_y",
        "hip_lost", "knee_lost", "ankle_lost",
        "force_N", "mass_kg",
        "dyno_status"
    ])

    vts_cap2, vw2, vh2, _ = open_camera(VTS_DEVICE)
    if vts_cap2:
        cv2.resizeWindow("VTS Preview", vw2, vh2)

    cap2 = cv2.VideoCapture(_TEMP_KNEE)
    print("Writing outputs...")

    fi = 0
    while True:
        ret, frame = cap2.read()
        if not ret or fi >= len(tracked_pts): break

        pts   = tracked_pts[fi]
        angle = tracked_angles[fi]
        lost  = tracked_lost[fi]
        confs = [0.0 if lost[i] else 1.0 for i in range(3)]

        # Wall-clock time for this frame (linear interpolation between
        # rec_start_time and rec_end_time, or None if not available)
        frame_time_s = fi / fps
        if rec_start_time is not None:
            frame_wall_time = rec_start_time + frame_time_s
        else:
            frame_wall_time = None

        # Match dynamometer sample
        if frame_wall_time is not None:
            force_N, mass_kg = dyno.get_sample_at(frame_wall_time)
        else:
            force_N, mass_kg = 0.0, 0.0

        # raw frame
        raw_writer.write(frame)
        cv2.imwrite(os.path.join(_KNEE_FRAMES_RAW, f"frame_{fi:06d}.png"), frame)

        # annotated frame
        annotated = draw_skeleton(frame.copy(), pts, confs, angle)
        ann_writer.write(annotated)
        cv2.imwrite(os.path.join(_KNEE_FRAMES_ANNOTATED, f"frame_{fi:06d}.png"), annotated)

        csv_wr.writerow([
            fi,
            round(frame_time_s, 4),
            round(frame_wall_time, 4) if frame_wall_time else "",
            round(angle, 2),
            round(float(pts[0][0]), 2), round(float(pts[0][1]), 2),   # hip_x,   hip_y
            round(float(pts[1][0]), 2), round(float(pts[1][1]), 2),   # knee_x,  knee_y
            round(float(pts[2][0]), 2), round(float(pts[2][1]), 2),   # ankle_x, ankle_y
            *[int(l) for l in lost],
            round(force_N, 4),
            round(mass_kg, 6),
            dyno.status
        ])

        if fi % DISPLAY_EVERY_N_FRAMES == 0:
            kdisp = annotated.copy()
            draw_hud_bar(kdisp,
                         [(f"WRITING OUTPUT {fi+1} / {len(tracked_pts)}", HUD_ACCENT, 0.65, 2)],
                         position="bottom")
            draw_force_overlay(kdisp, force_N, mass_kg)
            cv2.imshow("Main Cam", kdisp)
            if vts_cap2 and vts_cap2.isOpened():
                ret_v, vframe = vts_cap2.read()
                if ret_v:
                    vdisp = vframe.copy()
                    draw_vts_hud(vdisp, "WRITING", fi=fi, total=len(tracked_pts))
                    cv2.imshow("VTS Preview", vdisp)
            cv2.waitKey(1)

        fi += 1

    cap2.release()
    raw_writer.release()
    ann_writer.release()
    csv_file.close()
    if vts_cap2: vts_cap2.release()

    print("Knee outputs written.")
    print(f"  knee_raw       → {OUTPUT_KNEE_RAW}")
    print(f"  knee_annotated → {OUTPUT_KNEE_ANNOTATED}")
    print(f"  CSV            → {OUTPUT_CSV}")
    print(f"  frames/raw     → {_KNEE_FRAMES_RAW}")
    print(f"  frames/ann     → {_KNEE_FRAMES_ANNOTATED}")

    # ---- Launch Eulerian processing in background ----
    print("\n[EMM] Launching VTS Eulerian processing in background thread...")
    emm_thread = threading.Thread(
        target=eulerian_process_vts,
        args=(OUTPUT_VTS_RAW, OUTPUT_VTS_PROCESSED, _VTS_FRAMES_PROCESSED),
        daemon=False   # must NOT be daemon — process must stay alive until EMM finishes
    )
    emm_thread.start()
    return emm_thread


# ================================================================
# Main
# ================================================================

kfps, kw, kh, vfps, vw, vh, rec_start_time, rec_end_time = record_phase()
seed_clicks, first_gray = click_joints_phase(kw, kh)
emm_thread = track_and_write_phase(
    seed_clicks, first_gray, kw, kh, kfps,
    rec_start_time, rec_end_time
)

cv2.destroyAllWindows()
print("\nMain pipeline complete. Waiting for background Eulerian processing to finish...")
emm_thread.join()
dyno.stop()

print("\n=== ALL DONE ===")
print(f"Session timestamp : {_TS}")
print(f"Knee folder       : {_KNEE_FOLDER}")
print(f"VTS folder        : {_VTS_FOLDER}")
print(f"CSV               : {OUTPUT_CSV}")
print(f"Dyno status       : {dyno.status}")