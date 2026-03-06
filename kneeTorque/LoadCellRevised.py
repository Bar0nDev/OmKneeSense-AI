import sys
import time
import serial
import csv
from collections import deque
from datetime import datetime
from threading import Thread, Lock

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# ================= SETTINGS =================
SERIAL_PORT = "COM6"
BAUD = 115200
HISTORY = 500
GRAVITY = 9.80665
TARE_SAMPLES = 150
NOISE_THRESHOLD = 0.05
# ============================================

print("Opening serial port...")
ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

lock = Lock()

# ================= GLOBAL TARE =================
print("Global tare... unload device.")

tare_samples = []
while len(tare_samples) < TARE_SAMPLES:
    line = ser.readline().decode(errors="ignore").strip()
    parts = line.split(",")
    if len(parts) == 2:
        try:
            tare_samples.append(float(parts[1]))
        except:
            pass

tare_offset = sum(tare_samples) / len(tare_samples)
print(f"Global tare offset = {tare_offset:.3f}")

rep_zero_offset = 0.0
latest_raw_mass = 0.0

# ================= DATA STORAGE =================
mass_hist = deque([0]*HISTORY, maxlen=HISTORY)
force_hist = deque([0]*HISTORY, maxlen=HISTORY)

filename = f"force_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "raw_mass_g", "mass_kg", "force_N"])

running = True

# ================= SERIAL THREAD =================
def serial_reader():
    global running, latest_raw_mass

    while running:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 2:
                continue

            raw_mass = float(parts[1])

            with lock:
                latest_raw_mass = raw_mass
                corrected_mass = raw_mass - tare_offset - rep_zero_offset
                mass_kg = corrected_mass / 1000.0
                force_N = mass_kg * GRAVITY

                if force_N < NOISE_THRESHOLD:
                    force_N = 0.0
                    mass_kg = 0.0

            timestamp = time.time()
            csv_writer.writerow([timestamp, raw_mass, mass_kg, force_N])

            mass_hist.append(mass_kg)
            force_hist.append(force_N)

        except:
            pass

thread = Thread(target=serial_reader, daemon=True)
thread.start()

# ================= GUI =================
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
win.setWindowTitle("Knee Dynamometer - Force Monitor")
layout = QtWidgets.QVBoxLayout(win)

# Status Label
status_label = QtWidgets.QLabel("🟢 Global tare complete.\nPress SPACE when leg is back at rest to zero rep.")
status_label.setAlignment(QtCore.Qt.AlignCenter)
status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
layout.addWidget(status_label)

# Plot
plot_widget = pg.PlotWidget(title="Rep-Zeroed Force")
plot_widget.showGrid(x=True, y=True)
plot_widget.addLegend()
layout.addWidget(plot_widget)

curve_mass = plot_widget.plot(pen='y', name="Mass (kg)")
curve_force = plot_widget.plot(pen='c', name="Force (N)")

win.resize(900, 600)
win.show()

def update():
    curve_mass.setData(list(mass_hist))
    curve_force.setData(list(force_hist))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

# ================= KEY CONTROLS =================
def keyPressEvent(event):
    global rep_zero_offset, tare_offset

    if event.key() == QtCore.Qt.Key_Space:
        with lock:
            rep_zero_offset = latest_raw_mass - tare_offset
        status_label.setText("🔵 Rep zero set.\nPerform next repetition.")
        print("Rep zero set.")

    if event.key() == QtCore.Qt.Key_T:
        status_label.setText("🟡 Re-taring... unload device.")
        samples = []
        while len(samples) < TARE_SAMPLES:
            line = ser.readline().decode(errors="ignore").strip()
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    samples.append(float(parts[1]))
                except:
                    pass
        tare_offset = sum(samples) / len(samples)
        status_label.setText("🟢 Global tare complete.\nPress SPACE to zero next rep.")
        print("Global re-tare complete.")

win.keyPressEvent = keyPressEvent

# ================= CLEAN SHUTDOWN =================
def close():
    global running
    status_label.setText("🔴 Shutting down...")
    running = False
    thread.join(timeout=2)
    ser.close()
    csv_file.close()

app.aboutToQuit.connect(close)
sys.exit(app.exec())