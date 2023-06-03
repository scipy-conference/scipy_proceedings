#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Test the speed of rapidly updating multiple plot curves
"""

## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
app = pg.mkQApp("MultiPlot Speed Test")


colorSet = "viridis"
nPlots = 30
nSamples = 5000
height = 600
width = 1000
penWidth = 1

pg.setConfigOptions(background="w", foreground="k")

plot = pg.plot()
plot.setWindowTitle('PyQtGraph Example: MultiPlotSpeedTest')
plot.setLabel('bottom', 'Index', units='B')
plot.showAxis("left", False)


colorMap = pg.colormap.get(colorSet, source="matplotlib")
colors = colorMap.getLookupTable(nPts=nPlots)

curves = []
for idx in range(nPlots):
    curve = pg.PlotCurveItem(pen={"color": colors[idx], "width": penWidth})
    plot.addItem(curve)
    curve.setPos(0,idx*6)
    curves.append(curve)

plot.setYRange(0, nPlots*6)
plot.setXRange(0, nSamples)
plot.resize(width, height)

# rgn = pg.LinearRegionItem([nSamples/5.,nSamples/3.])
# plot.addItem(rgn)

data = np.random.normal(size=(nPlots*23,nSamples))
ptr = 0
lastTime = time()
fps = None
count = 0
def update():
    global curve, data, ptr, plot, lastTime, fps, nPlots, count
    count += 1

    for i in range(nPlots):
        curves[i].setData(data[(ptr+i)%data.shape[0]])

    ptr += nPlots
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    plot.setTitle(f'{nPlots} Lines with {nSamples} Points Each - %0.2f fps' % fps)
    # print(1_000 / fps)
    #app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    pg.mkQApp().exec_()
