# All matplotlib window code is modified from the following link:
# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html
import random
import numpy as np
from pyqtgraph.parametertree import Parameter, ParameterTree
import sys
import cv2 as cv
import pyqtgraph as pg
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from skimage.exposure import rescale_intensity
from utilitys.widgets import EasyWidget
import time
if QtCore.qVersion() >= "5.":
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from skimage import util, morphology as morph
from skimage import io
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure

app = pg.mkQApp()
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k')
UPDATE_FMT = '{lib} Update time (s): {updateTime:.5f}'

opts = [
    dict(name='kernel size', type='int', value=0, step=2, limits=[-101, 101]),
    dict(name='binarize', type='bool', value=False),
    dict(name='grayscale', type='bool', value=False),
    dict(name='strel shape', type='list',
         limits={'disk': morph.disk, 'square': morph.square}),
    dict(name='image size', type='int', value=1000, step=1000, limits=[10,5000])
]
param = Parameter.create(name='Options', type='group', children=opts)
baseUrl = 'https://picsum.photos/'
img = newImg = None
tree = ParameterTree()
tree.setParameters(param)

def changeImg():
    global img
    useUrl = baseUrl + str(param['image size'])
    img = io.imread(useUrl)
changeImg()
param.child('image size').sigValueChanged.connect(changeImg)


def applyOp():
    global newImg
    ksize = param['kernel size']
    op = cv.MORPH_DILATE
    if ksize == 0:
        newImg = img.copy()
    else:
        if ksize < 0:
            op = cv.MORPH_ERODE
            ksize = -ksize
        newImg = cv.morphologyEx(img, op, param['strel shape'](ksize))
    if param['grayscale']:
        newImg = newImg.mean(2)
    if param['binarize']:
        newImg = newImg > newImg.mean()
param.sigTreeStateChanged.connect(applyOp)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots()
        self.ax.set_axis_off()
        self.updateTime = 0.0
        self.addToolBar(NavigationToolbar(self.canvas, self))

        self.setCentralWidget(self.canvas)
        self.figure.tight_layout(pad=2)
        self.figure.suptitle('Morphological Dilation')
        param.sigTreeStateChanged.connect(self.updateImage)

    def updateImage(self):
        global newImg
        toPlot = rescale_intensity(newImg, out_range='uint8')
        start = time.perf_counter()
        self.ax.clear()
        self.ax.imshow(toPlot)
        self.canvas.draw()
        self.updateTime = time.perf_counter() - start
        self.statusBar().showMessage(UPDATE_FMT.format(lib='Matplotlib', updateTime=self.updateTime))

class MyPlotWidget(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        pw = self.pw = pg.PlotWidget()
        param.sigTreeStateChanged.connect(self.updateImage)
        self.imgItem = pg.ImageItem()
        pw.addItem(self.imgItem)
        pw.getViewBox().invertY()
        pw.getViewBox().setAspectLocked()
        self.updateTime = 0.0
        self.setCentralWidget(pw)
        pw.plotItem.setTitle('Morphological Dilation')

    def updateImage(self):
        global newImg
        start = time.perf_counter()
        self.imgItem.setImage(newImg)
        self.updateTime = time.perf_counter() - start
        self.statusBar().showMessage(UPDATE_FMT.format(lib='PyQtGraph', updateTime=self.updateTime))

if __name__ == "__main__":
    w1 = MainWindow()
    w2 = MyPlotWidget()
    lbl = QtWidgets.QLabel()
    win = EasyWidget.buildMainWin([w1, w2, [tree, lbl]], layout='H')
    param.sigTreeStateChanged.connect(
        lambda: lbl.setText(f'PyQtGraph speedup: {w1.updateTime/w2.updateTime:3.2f}x'))
    param.sigTreeStateChanged.emit(param, [])
    win.show()
    sys.exit(app.exec_())