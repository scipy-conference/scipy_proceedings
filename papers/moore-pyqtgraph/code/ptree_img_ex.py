# All matplotlib window code is modified from the following link:
# https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html
import random
import numpy as np
from pyqtgraph.parametertree import Parameter, ParameterTree
import sys
import cv2 as cv
import pyqtgraph as pg
from utilitys.widgets import EasyWidget
from skimage import morphology as morph
from skimage import io, data
from PyQt5 import  QtWidgets

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
img = newImg = data.camera()
tree = ParameterTree()
tree.setParameters(param)

def changeImg():
    global img
    useUrl = baseUrl + str(param['image size'])
    img = io.imread(useUrl)
# changeImg()
# param.child('image size').sigValueChanged.connect(changeImg)

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

class MyPlotWidget(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        pw = self.pw = pg.PlotWidget()
        param.sigTreeStateChanged.connect(self.updateImage)
        self.imgItem = pg.ImageItem()
        pw.addItem(self.imgItem)
        pw.getViewBox().invertY()
        pw.getViewBox().setAspectLocked()
        self.setCentralWidget(pw)
        pw.plotItem.setTitle('Morphological Dilation')

    def updateImage(self):
        self.imgItem.setImage(newImg)

if __name__ == "__main__":
    plotWin = MyPlotWidget()
    lbl = QtWidgets.QLabel()
    win = EasyWidget.buildMainWin([plotWin, tree], layout='H')
    param.sigTreeStateChanged.emit(param, [])
    win.show()
    sys.exit(app.exec_())