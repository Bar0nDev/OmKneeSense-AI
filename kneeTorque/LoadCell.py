import sys
import time
import serial
import csv
from collections import deque
from datetime import datetime
from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# ================= SETTINGS =================
SERIAL_PORT = "COM6"
BAUD = 115200
HISTORY = 500
GRAVITY = 9.80665
# ============================================

ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

mass_hist = deque([0]*HISTORY, maxlen=HISTORY)
force_hist = deque([0]*HISTORY, maxlen=HISTORY)

# CSV logging
filename = f"force_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "mass_kg", "force_N"])

running = True

# -------- Serial Thread --------
def serial_reader():
    global running
    while running:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 2:
                continue

            mass_kg = float(parts[1]) / 1000.0
            force_N = mass_kg  * GRAVITY

            timestamp = time.time()
            csv_writer.writerow([timestamp, mass_kg, force_N])

            mass_hist.append(mass_kg)
            force_hist.append(force_N)

        except:
            pass

thread = Thread(target=serial_reader)
thread.start()

# -------- PyQtGraph App --------
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Force Monitor")
win.resize(900, 500)

plot = win.addPlot(title="Mass (kg) and Force (N)")
plot.showGrid(x=True, y=True)

# Add legend
legend = plot.addLegend()

curve_mass = plot.plot(pen='y', name="Mass (kg)")
curve_force = plot.plot(pen='c', name="Force (N)")

def update():
    curve_mass.setData(list(mass_hist))
    curve_force.setData(list(force_hist))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)  # 50 FPS GUI refresh

def close():
    global running
    running = False
    thread.join()
    ser.close()
    csv_file.close()

app.aboutToQuit.connect(close)
sys.exit(app.exec())
