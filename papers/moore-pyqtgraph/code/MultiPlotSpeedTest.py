"""
Test the speed of rapidly updating multiple plot curves
"""
import argparse
import itertools

import numpy as np
from pyqtgraph.examples.utils import FrameCounter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', default=float('inf'), type=float,
    help="Number of iterations to run before exiting"
)
parser.add_argument('--duration', default=float('inf'), type=float,
    help="Duration to run the benchmark for before exiting"
)
args = parser.parse_args()

iterations_counter = itertools.count()
# pg.setConfigOptions(useOpenGL=True)
app = pg.mkQApp("MultiPlot Speed Test")

plot = pg.plot()
plot.setWindowTitle('pyqtgraph example: MultiPlotSpeedTest')
# plot.setLabel('bottom', 'Index', units='B')

nPlots = 100
nSamples = 1_00
curves = []
for idx in range(nPlots):
    # with downsampling
    curve = pg.PlotDataItem(pen=({'color': (idx, nPlots*1.3), 'width': 1}), skipFiniteCheck=True)
    # without downsampling

    plot.addItem(curve)
    curve.setPos(0,idx*6)
    curves.append(curve)

plot.setYRange(0, nPlots*6)
plot.setXRange(0, nSamples)
plot.resize(600,900)

# will raise an error if using PlotCurveItem's directly
# plot.getPlotItem().setDownsampling(ds=True, auto=True, mode="peak")

ms_readings = []

data = np.random.normal(size=(nPlots*23,nSamples))
ptr = 0
start = perf_counter()
def update():
    global ptr
    if next(iterations_counter) > args.iterations:
        timer.stop()
        app.quit()
        print(np.mean(ms_readings))
        return None
    elif perf_counter() - start > args.duration:
        timer.stop()
        app.quit()
        print(np.mean(ms_readings))
        return None
    for i in range(nPlots):
        curves[i].setData(data[(ptr+i)%data.shape[0]])

    ptr += nPlots
    framecnt.update()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps: plot.setTitle(f'{1000 / fps:.3f} ms'))
framecnt.sigFpsUpdate.connect(lambda fps: ms_readings.append(1000 / fps))
if __name__ == '__main__':
    pg.exec()
