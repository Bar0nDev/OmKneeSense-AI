"""
joint_editor.py  -  Standalone Joint Dot Adjustment Editor
===========================================================
Run:  python joint_editor.py
      python joint_editor.py --root "C:/path/to/output"

Requires: opencv-python, numpy, scipy
"""

import os
import sys
import csv
import shutil
import argparse
from datetime import datetime

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# ================================================================
# CONFIG  -- mirror the paths in ultimateDataGathering.py
# ================================================================
DEFAULT_OUT_ROOT = os.path.join(".", "kneeAngle", "updated", "output")
KNEE_ROOT_NAME   = "knee"
CSV_ROOT_NAME    = "outputCSV"

JOINT_NAMES  = ["HIP", "KNEE", "ANKLE"]
JOINT_COLORS = [(0, 200, 255), (0, 255, 100), (80, 80, 255)]   # BGR

# HUD palette
HUD_BG     = (15,  15,  15)
HUD_ACCENT = (0,  210, 255)
HUD_DIM    = (120, 120, 120)
HUD_WHITE  = (230, 230, 230)
HUD_GREEN  = (60,  220, 100)
HUD_YELLOW = (0,  200, 180)
HUD_ORANGE = (30,  140, 255)
HUD_RED    = (60,   60, 220)

MIN_TRACK_CONFIDENCE = 0.3

# Optical-flow params (same as main script)
LK_PARAMS = dict(
    winSize  = (41, 41),
    maxLevel = 4,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
FB_ERROR_THRESHOLD = 3.0

# Butterworth low-pass cutoff for angle smoothing on export
SMOOTH_CUTOFF_HZ = 6.0

# ================================================================
# Geometry
# ================================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc  = a - b, c - b
    cos     = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def smooth_angles(angles, fps, cutoff_hz=SMOOTH_CUTOFF_HZ):
    nyq = 0.5 * fps
    if cutoff_hz >= nyq or len(angles) < 9:
        return list(angles)
    b, a = butter(4, cutoff_hz / nyq, btype="low")
    return list(filtfilt(b, a, angles))

# ================================================================
# Drawing
# ================================================================

def draw_skeleton(frame, pts, lost):
    all_ok     = not any(lost)
    line_color = (0, 230, 120) if all_ok else (0, 165, 255)
    cv2.line(frame, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), line_color, 3)
    cv2.line(frame, tuple(pts[1].astype(int)), tuple(pts[2].astype(int)), line_color, 3)
    for i, (pt, lf) in enumerate(zip(pts, lost)):
        p = tuple(pt.astype(int))
        cv2.circle(frame, p, 9, JOINT_COLORS[i] if not lf else (60, 60, 200), -1)
        cv2.circle(frame, p, 9, (255, 255, 255), 2)
        cv2.putText(frame, JOINT_NAMES[i], (p[0]+12, p[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, JOINT_COLORS[i], 2)
    return frame


def draw_angle(frame, pts, angle):
    knee = tuple(pts[1].astype(int))
    cv2.putText(frame, f"{angle:.1f} deg", (knee[0]+20, knee[1]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 0), 2)
    return frame


def draw_hud_bar(frame, lines, position="bottom", alpha=0.72):
    h, w    = frame.shape[:2]
    line_h  = 30
    pad     = 12
    bar_h   = pad * 2 + line_h * len(lines)
    y0      = (h - bar_h) if position == "bottom" else 0
    y1      = h            if position == "bottom" else bar_h
    roi     = frame[y0:y1, 0:w]
    overlay = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, HUD_BG), alpha, 0)
    frame[y0:y1, 0:w] = overlay
    accent_y = y0 if position == "bottom" else y1
    cv2.line(frame, (0, accent_y), (w, accent_y), HUD_ACCENT, 2)
    for i, (text, color, scale, thick) in enumerate(lines):
        ty = y0 + pad + line_h * i + 20
        cv2.putText(frame, text, (pad + 6, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return frame


def draw_badge(frame, text, color, position="top-right"):
    h, w = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 10
    bw, bh = tw + pad * 2, th + pad * 2
    x0 = w - bw - 10 if "right" in position else 10
    y0 = 10
    roi     = frame[y0:y0+bh, x0:x0+bw]
    overlay = cv2.addWeighted(roi, 0.15, np.full_like(roi, color), 0.85, 0)
    frame[y0:y0+bh, x0:x0+bw] = overlay
    cv2.rectangle(frame, (x0, y0), (x0+bw, y0+bh), color, 2)
    cv2.putText(frame, text, (x0+pad, y0+bh-pad),
                font, scale, HUD_WHITE, thick, cv2.LINE_AA)
    return frame


def draw_timeline(frame, fi, total, adjusted_frames):
    """Thin strip at very bottom: orange ticks = adjusted, white line = playhead."""
    h, w    = frame.shape[:2]
    strip_h = 10
    y0      = h - strip_h
    cv2.rectangle(frame, (0, y0), (w, h), (25, 25, 25), -1)
    for afi in adjusted_frames:
        x = int(afi / max(total - 1, 1) * (w - 1))
        cv2.line(frame, (x, y0), (x, h), HUD_ORANGE, 2)
    px = int(fi / max(total - 1, 1) * (w - 1))
    cv2.line(frame, (px, y0 - 2), (px, h), (255, 255, 255), 2)
    return frame

# ================================================================
# Session scanning
# ================================================================

def scan_sessions(out_root):
    knee_root = os.path.join(out_root, KNEE_ROOT_NAME)
    csv_root  = os.path.join(out_root, CSV_ROOT_NAME)
    if not os.path.isdir(knee_root):
        return []

    sessions = []
    for name in sorted(os.listdir(knee_root), reverse=True):
        folder = os.path.join(knee_root, name)
        if not os.path.isdir(folder):
            continue
        ts         = name.replace("knee_", "")
        raw_video  = os.path.join(folder, "knee_raw.mp4")
        ann_video  = os.path.join(folder, "knee_annotated.mp4")
        frames_ann = os.path.join(folder, "frames", "annotated")

        csv_path = None
        if os.path.isdir(csv_root):
            for f in sorted(os.listdir(csv_root)):
                if ts in f and f.endswith(".csv"):
                    csv_path = os.path.join(csv_root, f)
                    break

        if not os.path.isfile(raw_video) or not csv_path:
            continue

        has_xy = False
        try:
            with open(csv_path, newline="") as fh:
                headers = csv.DictReader(fh).fieldnames or []
                has_xy  = "hip_x" in headers
        except Exception:
            pass

        sessions.append({
            "ts":         ts,
            "folder":     folder,
            "raw_video":  raw_video,
            "ann_video":  ann_video,
            "frames_ann": frames_ann,
            "csv_path":   csv_path,
            "has_xy":     has_xy,
        })
    return sessions


def pick_session(sessions):
    print("\n" + "="*64)
    print("  JOINT EDITOR  -  Session Selector")
    print("="*64)
    if not sessions:
        print("  No sessions found.")
        print(f"  Scanning: {os.path.abspath(DEFAULT_OUT_ROOT)}")
        return None

    print(f"  {len(sessions)} session(s) found:\n")
    for i, s in enumerate(sessions):
        try:
            dt    = datetime.strptime(s["ts"], "%Y%m%d_%H%M%S")
            label = dt.strftime("%Y-%m-%d  %H:%M:%S")
        except ValueError:
            label = s["ts"]
        xy_tag = "" if s["has_xy"] else "  [NO XY - needs updated main script]"
        print(f"  [{i+1:2d}]  {label}{xy_tag}")
        print(f"        {os.path.basename(s['csv_path'])}")
        print()

    while True:
        raw = input("  Enter session number (or Q to quit): ").strip()
        if raw.lower() == "q":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]
        except ValueError:
            pass
        print(f"  Please enter 1 - {len(sessions)}.")

# ================================================================
# CSV I/O
# ================================================================

def load_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows       = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    has_xy = "hip_x" in fieldnames
    pts, lost, angles = [], [], []

    for row in rows:
        angles.append(float(row.get("angle_deg", 0.0)))
        lost.append([
            bool(int(row.get("hip_lost",   0))),
            bool(int(row.get("knee_lost",  0))),
            bool(int(row.get("ankle_lost", 0))),
        ])
        if has_xy:
            pts.append(np.array([
                [float(row["hip_x"]),   float(row["hip_y"])],
                [float(row["knee_x"]),  float(row["knee_y"])],
                [float(row["ankle_x"]), float(row["ankle_y"])],
            ], dtype=np.float32))
        else:
            pts.append(None)

    fps = 30.0
    if len(rows) > 1:
        try:
            t0  = float(rows[0]["timestamp_s"])
            t1  = float(rows[1]["timestamp_s"])
            fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 30.0
        except (KeyError, ValueError, ZeroDivisionError):
            pass

    return rows, pts, lost, angles, fps, fieldnames, has_xy


def write_csv(csv_path, rows, pts, lost, angles_smoothed, fieldnames):
    xy_cols = ["hip_x", "hip_y", "knee_x", "knee_y", "ankle_x", "ankle_y"]
    for col in xy_cols:
        if col not in fieldnames:
            idx = fieldnames.index("angle_deg") + 1
            fieldnames.insert(idx, col)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            r = dict(row)
            r["angle_deg"]  = round(angles_smoothed[i], 4)
            r["hip_x"]      = round(float(pts[i][0][0]), 2)
            r["hip_y"]      = round(float(pts[i][0][1]), 2)
            r["knee_x"]     = round(float(pts[i][1][0]), 2)
            r["knee_y"]     = round(float(pts[i][1][1]), 2)
            r["ankle_x"]    = round(float(pts[i][2][0]), 2)
            r["ankle_y"]    = round(float(pts[i][2][1]), 2)
            r["hip_lost"]   = int(lost[i][0])
            r["knee_lost"]  = int(lost[i][1])
            r["ankle_lost"] = int(lost[i][2])
            writer.writerow(r)

# ================================================================
# Optical flow re-propagation  (always runs to the end of the video)
# ================================================================

def repropagate(frames_gray, pts, lost, from_fi):
    """
    Re-run forward LK from from_fi onward all the way to the last frame.
    No early stopping -- a confidently-wrong tracker would pass the FB
    check anyway, so we always re-propagate everything after the edit.
    Mutates pts and lost in-place.
    """
    total        = len(pts)
    prev_gray    = frames_gray[from_fi]
    prev_pts_arr = pts[from_fi].reshape(-1, 1, 2).astype(np.float32)

    for fi in range(from_fi + 1, total):
        gray = frames_gray[fi]
        fwd, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_arr, None, **LK_PARAMS)
        bwd, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, fwd,          None, **LK_PARAMS)

        new_pts  = pts[fi].copy()
        new_lost = [False, False, False]

        for j in range(3):
            both_ok = (st_fwd[j][0] == 1) and (st_bwd[j][0] == 1)
            if both_ok:
                fb_err = np.linalg.norm(prev_pts_arr[j][0] - bwd[j][0])
                if fb_err < FB_ERROR_THRESHOLD:
                    new_pts[j]  = fwd[j][0]
                    new_lost[j] = False
                else:
                    new_lost[j] = True
            else:
                new_lost[j] = True

        pts[fi]  = new_pts
        lost[fi] = new_lost

        prev_gray    = gray
        prev_pts_arr = new_pts.reshape(-1, 1, 2).astype(np.float32)

# ================================================================
# Editor class
# ================================================================

WIN = "Joint Adjustment Editor"

class Editor:
    def __init__(self, session):
        self.session = session
        self.valid   = False

        print(f"\n[EDITOR] Loading CSV ...")
        rows, pts, lost, angles, fps, fieldnames, has_xy = load_csv(session["csv_path"])

        if not has_xy:
            print("\n  WARNING: This CSV has no XY columns.")
            print("  Re-run the updated ultimateDataGathering.py to capture a new session")
            print("  with XY data, then come back to edit it here.")
            input("\n  Press ENTER to return to session list.")
            return

        self.valid      = True
        self.rows       = rows
        self.pts        = pts
        self.lost       = lost
        self.angles     = angles
        self.fps        = fps
        self.fieldnames = fieldnames
        self.total      = len(rows)

        for i in range(self.total):
            self.angles[i] = calculate_angle(pts[i][0], pts[i][1], pts[i][2])

        print("[EDITOR] Loading raw video ...")
        self.frames, self.frames_gray = self._load_video(session["raw_video"])

        if len(self.frames) < self.total:
            print(f"  WARNING: video has {len(self.frames)} frames but CSV has {self.total}.")
            self.total  = len(self.frames)
            self.rows   = self.rows[:self.total]
            self.pts    = self.pts[:self.total]
            self.lost   = self.lost[:self.total]
            self.angles = self.angles[:self.total]

        print(f"[EDITOR] Ready  |  {self.total} frames  |  {self.fps:.1f} fps\n")

        self.fi            = 0
        self.selected      = 1
        self.adjusted      = set()
        self.undo_stack    = []
        self.jump_mode     = False
        self.jump_buf      = ""
        self.pending_click = None

    # ------------------------------------------------------------------
    def _load_video(self, path):
        cap    = cv2.VideoCapture(path)
        frames = []
        grays  = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
            grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        cap.release()
        return frames, grays

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_click = (x, y)

    def _push_undo(self):
        self.undo_stack.append((
            self.fi,
            [p.copy() for p in self.pts],
            [list(l) for l in self.lost],
            list(self.angles),
        ))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _undo(self):
        if not self.undo_stack:
            print("[EDITOR] Nothing to undo.")
            return
        snap_fi, snap_pts, snap_lost, snap_angles = self.undo_stack.pop()
        self.pts    = snap_pts
        self.lost   = snap_lost
        self.angles = snap_angles
        self.adjusted.discard(snap_fi)
        self.fi = snap_fi
        print(f"[EDITOR] Undo -- reverted to before frame {snap_fi}.")

    # ------------------------------------------------------------------
    def _apply_click(self, x, y):
        fi = self.fi
        self._push_undo()

        self.pts[fi][self.selected]  = np.array([x, y], dtype=np.float32)
        self.lost[fi][self.selected] = False
        self.angles[fi] = calculate_angle(self.pts[fi][0], self.pts[fi][1], self.pts[fi][2])
        self.adjusted.add(fi)

        remaining = self.total - fi - 1
        print(f"[EDITOR] Frame {fi:>5}: {JOINT_NAMES[self.selected]:5s} -> ({x:4d},{y:4d})  "
              f"angle={self.angles[fi]:.1f}  re-propagating {remaining} frames...",
              end="", flush=True)

        repropagate(self.frames_gray, self.pts, self.lost, fi)

        for i in range(fi + 1, self.total):
            self.angles[i] = calculate_angle(self.pts[i][0], self.pts[i][1], self.pts[i][2])

        print("  done.")

    # ------------------------------------------------------------------
    def _render(self):
        fi    = self.fi
        frame = self.frames[fi].copy()
        pts   = self.pts[fi]
        lost  = self.lost[fi]
        angle = self.angles[fi]

        draw_skeleton(frame, pts, lost)
        draw_angle(frame, pts, angle)

        sx, sy = int(pts[self.selected][0]), int(pts[self.selected][1])
        cv2.circle(frame, (sx, sy), 18, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"< {JOINT_NAMES[self.selected]}",
                    (sx + 22, sy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

        jstrip = "   ".join(
            f"[{i+1}]{'*' if i == self.selected else ' '}{JOINT_NAMES[i]}"
            for i in range(3)
        )
        draw_hud_bar(frame, [
            (f"Frame {fi+1:>5} / {self.total}     {jstrip}", HUD_ACCENT, 0.50, 1),
            ("1/2/3 joint   LMB move   U undo   S save   Q quit   "
             "A/D step1   ,/. step10   J jump",
             HUD_DIM, 0.38, 1),
        ], position="bottom")

        if fi in self.adjusted:
            draw_badge(frame, "ADJUSTED", HUD_YELLOW, "top-right")

        if self.jump_mode:
            draw_hud_bar(frame,
                         [(f"Jump to frame: {self.jump_buf}_", HUD_ORANGE, 0.65, 2)],
                         position="top")

        draw_timeline(frame, fi, self.total, self.adjusted)
        return frame

    # ------------------------------------------------------------------
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        h0, w0 = self.frames[0].shape[:2]
        cv2.resizeWindow(WIN, w0, h0)
        cv2.setMouseCallback(WIN, self._mouse_cb)

        print("Controls:  1/2/3=joint   LMB=move joint   U=undo   S=save+exit   Q=quit")
        print("           A/D = step 1 frame    ,/. = step 10    J = jump\n")

        while True:
            self.fi = max(0, min(self.fi, self.total - 1))
            cv2.imshow(WIN, self._render())

            if self.pending_click and not self.jump_mode:
                self._apply_click(*self.pending_click)
                self.pending_click = None
                continue

            key = cv2.waitKey(20) & 0xFF

            if self.jump_mode:
                if key == 13:                      # ENTER
                    try:
                        self.fi = max(0, min(int(self.jump_buf) - 1, self.total - 1))
                    except ValueError:
                        pass
                    self.jump_mode = False
                    self.jump_buf  = ""
                elif key == 8:                     # BACKSPACE
                    self.jump_buf = self.jump_buf[:-1]
                elif key == 27:                    # ESC cancel
                    self.jump_mode = False
                    self.jump_buf  = ""
                elif 48 <= key <= 57:              # digit
                    self.jump_buf += chr(key)
                continue

            if   key == ord('a'):     self.fi = max(0, self.fi - 1)       # prev frame
            elif key == ord('d'):     self.fi = min(self.total - 1, self.fi + 1)  # next frame
            elif key == ord(','):     self.fi = max(0, self.fi - 10)       # prev 10
            elif key == ord('.'):     self.fi = min(self.total - 1, self.fi + 10) # next 10
            elif key == ord('j'):
                self.jump_mode = True
                self.jump_buf  = ""
            elif key == ord('1'):     self.selected = 0
            elif key == ord('2'):     self.selected = 1
            elif key == ord('3'):     self.selected = 2
            elif key == ord('u'):     self._undo()
            elif key == ord('s'):
                cv2.destroyWindow(WIN)
                self._save()
                return True
            elif key in (ord('q'), 27):
                ans = input("\n  Quit without saving? [y/N]: ").strip().lower()
                if ans == "y":
                    cv2.destroyWindow(WIN)
                    return False

    # ------------------------------------------------------------------
    def _save(self):
        csv_path = self.session["csv_path"]
        ann_vid  = self.session["ann_video"]
        frm_dir  = self.session["frames_ann"]

        bak = csv_path.replace(".csv", "_backup.csv")
        if not os.path.isfile(bak):
            shutil.copy2(csv_path, bak)
            print(f"[SAVE] CSV backup       -> {bak}")

        print(f"[SAVE] Smoothing angles (cutoff {SMOOTH_CUTOFF_HZ} Hz, fps {self.fps:.1f}) ...")
        smoothed = smooth_angles(self.angles, self.fps)

        write_csv(csv_path, self.rows, self.pts, self.lost, smoothed, list(self.fieldnames))
        print(f"[SAVE] CSV written      -> {csv_path}")

        h, w = self.frames[0].shape[:2]
        os.makedirs(frm_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw     = cv2.VideoWriter(ann_vid, fourcc, self.fps, (w, h))

        print(f"[SAVE] Rendering {self.total} annotated frames ...")
        for fi in range(self.total):
            frame = self.frames[fi].copy()
            draw_skeleton(frame, self.pts[fi], self.lost[fi])
            draw_angle(frame, self.pts[fi], smoothed[fi])
            vw.write(frame)
            cv2.imwrite(os.path.join(frm_dir, f"frame_{fi:06d}.png"), frame)
            if fi % 150 == 0:
                print(f"  {fi+1}/{self.total}")

        vw.release()
        print(f"[SAVE] Annotated video  -> {ann_vid}")
        print(f"[SAVE] Annotated frames -> {frm_dir}")
        print(f"[SAVE] Done.  {len(self.adjusted)} frame(s) manually adjusted.")

# ================================================================
# Entry point
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone joint dot adjustment editor.")
    parser.add_argument("--root", default=DEFAULT_OUT_ROOT,
                        help="Output root folder (default: %(default)s)")
    args = parser.parse_args()

    out_root = args.root
    print(f"\nScanning: {os.path.abspath(out_root)}")
    sessions = scan_sessions(out_root)

    while True:
        session = pick_session(sessions)
        if session is None:
            break

        editor = Editor(session)
        if not editor.valid:
            continue

        saved = editor.run()
        print("\nSaved." if saved else "No changes saved.")

        again = input("\nEdit another session? [y/N]: ").strip().lower()
        if again != "y":
            break

    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()