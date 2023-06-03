import pyqtgrpah as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from pyqtgraph.ptime import time

app = pg.mkQApp("Line Plots Benchmark")


plot = pg.plot()
plot.setWindowTitle("MultiPlotSpeedTest")
plot.setLabel('bottom', "Index", units="B")


nPlotsRange = list(range(1, 501, 100))
nSamplesRange = list(range(100, 1001, 100))

results = np.empty((len(nSamplesRange), len(nPlotsRange)))



def getData():
    global x, y, ptr, lastTime, fps, count, nPlots,
    for y, nPlots in enumerate(nPlotsRange):
        for x, nSamples in enumerate(nSamplesRange):
            curves = []

            for idx in range(nPlots):
                curve = pg.PlotCurveItem(pen=(idx, nPlots * 1.3))
                plot.addItem(curve)
                curve.setPos(0, idx * 6)
                curves.append(curve)

            plot.setYRange(0, nPlots * 6)
            plot.setXRange(0, nSamples)
            plot.resize(600, 900)

            data = np.random.normal(size=(nPlots * 23, nSamples))
            ptr = 0
            lastTime = time()
            fps = None
            count = 0
            yield data



def update():
    global curve, data, ptr, plot, lastTime, fps, nPlots, count, x, y
    count += 1

    for i in range(nPlots):
        curves[i].setData(data[(ptr + i) % data.shape[0]])
    
    ptr += nPlots
    now = time()
    dt = now - lastTime
    lastTime = now

    if fps is None:
        fps = 1.0 / dt
    else:
        s = np.clip(dt * 3., 0, 1)
        fps = fps * (1 - s) + (1.0 / dt) * s


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


if __name__ == "__main__":
    pg.mkQApp().exec_()