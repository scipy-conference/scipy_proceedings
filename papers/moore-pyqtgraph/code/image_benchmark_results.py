import numpy as np
import pandas as pd
import pyqtgraph as pg

from pyqtgraph.Qt import QtGui, QtWidgets, QtCore


app = pg.mkQApp("Video Speed Test Results")

pg.setConfigOptions(antialias=True, background="w", foreground="k")

# namipulate the 
data = pd.read_csv("image_performance.csv")
data = data.melt(
    id_vars=["size", "acceleration", "use_levels", "dtype"],
    var_name="lut",
    value_name="time"
)
data = data.assign(
    pixels=(
        data["size"]
        .str
        .strip("()")
        .str
        .split(",", expand=True)
        .apply(lambda x: int(x[0]) * int(x[1]), axis=1)
    )
)


penWidth = 3
legendFontSize = "10pt"
titleFontSize = "12pt"
axisFontSize = "10pt"
fontL = QtGui.QFont("Arial")
fontL.setPointSize(14)

fontM = QtGui.QFont("Arial")
fontM.setPointSize(12)

gray = "#757575"
blue = "#41A7DC"
purple = "#B65FD3"
nvidia_green = "#76B900"
numba_blue = (57, 150, 215)
fillColor = "#00004040"
blazing = "#ff6961"


pen_cupy_yesLevels_noLUT = pg.mkPen(nvidia_green, width=penWidth)
pen_numpy_yesLevels_noLUT = pg.mkPen(purple, width=penWidth)
pen_numba_yesLevels_noLUT = pg.mkPen(numba_blue, width=penWidth)

pen_numpy_yesLevels_uint16LUT = pg.mkPen(purple, width=penWidth, style=pg.QtCore.Qt.PenStyle.DashLine)
pen_numba_yesLevels_uint16LUT = pg.mkPen(numba_blue, width=penWidth, style=pg.QtCore.Qt.PenStyle.DashLine)
pen_cupy_yesLevels_uint16LUT = pg.mkPen(nvidia_green, width=penWidth, style=QtCore.Qt.PenStyle.DashLine)

pen_blazing = pg.mkPen(blazing, width=penWidth)

infiniteLineColorSetName = "inferno"
infiniteLineColorMap = pg.colormap.get(infiniteLineColorSetName, source='matplotlib')

# breakpoint()

def getSeries(df):
    series = {}
    series["cupy_yesLevels_noLUT"] = df[(df["acceleration"] == "cupy") & (df["use_levels"]) & (df["lut"] == "no LUT")]["time"].to_numpy() * 1_000
    series["cupy_yesLevels_uint16LUT"] = df[(df["acceleration"] == "cupy") & (df["use_levels"]) & (df["lut"] == "uint16-lut")]["time"].to_numpy() * 1_000

    # series["cupy_noLevels_noLUT"] = df[(df["acceleration"] == "cupy") & (df["use_levels"] == False) & (df["lut"] == "no LUT")]["time"].to_numpy() * 1_000
    # series["cupy_noLevels_uint16LUT"] = df[(df["acceleration"] == "cupy") & (df["use_levels"] == False) & (df["lut"] == "uint16-lut")]["time"].to_numpy() * 1_000

    series["numpy_yesLevels_noLUT"] = df[(df["acceleration"] == "numpy") & (df["use_levels"]) & (df["lut"] == "no LUT")]["time"].to_numpy() * 1_000
    series["numpy_yesLevels_uint16LUT"] = df[(df["acceleration"] == "numpy") & (df["use_levels"]) & (df["lut"] == "uint16-lut")]["time"].to_numpy() * 1_000

    series["numpy_noLevels_noLUT"] = df[(df["acceleration"] == "numpy") & (df["use_levels"] == False) & (df["lut"] == "no LUT")]["time"].to_numpy() * 1_000
    # series["numpy_noLevels_uint16LUT"] = df[(df["acceleration"] == "numpy") & (df["use_levels"] == False) & (df["lut"] == "uint16-lut")]["time"].to_numpy() * 1_000

    series["numba_yesLevels_noLUT"] = df[(df["acceleration"] == "numba") & (df["use_levels"]) & (df["lut"] == "no LUT")]["time"].to_numpy() * 1_000
    series["numba_yesLevels_uint16LUT"] = df[(df["acceleration"] == "numba") & (df["use_levels"]) & (df["lut"] == "uint16-lut")]["time"].to_numpy() * 1_000

    # series["numba_noLevels_noLUT"] = df[(df["acceleration"] == "numba") & (df["use_levels"] == False) & (df["lut"] == "no LUT")]["time"].to_numpy() * 1_000
    # series["numba_noLevels_uint16LUT"] = df[(df["acceleration"] == "numba") & (df["use_levels"] == False) & (df["lut"] == "uint16-lut")]["time"].to_numpy() * 1_000


    return series

win = pg.GraphicsLayoutWidget()
x = np.square([256, 512, 1024, 2048, 3072, 4096])

uint8Data = getSeries(data.loc[data["dtype"] == "uint8"])
uint16Data = getSeries(data.loc[data["dtype"] == "uint16"])
floatData = getSeries(data.loc[data["dtype"] == "float32"])

floatPlot = win.addPlot(0,0, 1,1)
floatPlot.setFixedWidth(290)

uint16Plot = win.addPlot(0,2, 1,1)
uint16Plot.setFixedWidth(270)

print( win.ci.layout) 
win.ci.layout.setColumnFixedWidth(1,1)
win.ci.layout.setColumnFixedWidth(1,1)

for plot in (floatPlot, uint16Plot):
    plot.disableAutoRange()
    plot.setYRange( 0.1, 200, padding=0 )
    plot.titleLabel.setFont( fontL )
    plot.axes["bottom"]["item"].enableAutoSIPrefix(False)
    for loc in ('left','right','top','bottom'):
        ax = plot.getAxis(loc)
        ax.setStyle( tickFont=fontM )
        if loc in('right', 'top'): ax.setStyle( showValues=False )
        ax.label.setFont( fontM )
        ax.show()

floatPlot.setTitle("Float format performance", size=titleFontSize)


floatLegend = floatPlot.addLegend(brush="w", pen="k", verSpacing=-10, offset=(-5,-2), size=(10,10))
floatLegend.setLabelTextSize(legendFontSize)
noLUTcupy = floatPlot.plot(
    x=x,
    y=floatData["cupy_yesLevels_noLUT"],
    name="CuPy - no LUT",
    pen=pen_cupy_yesLevels_noLUT
)

yesLUTcupy = floatPlot.plot(
    x=x,
    y=floatData["cupy_yesLevels_uint16LUT"],
    name="(with LUT)",
    pen=pen_cupy_yesLevels_uint16LUT
)

noLUTnumba = floatPlot.plot(
    x=x,
    y=floatData["numba_yesLevels_noLUT"],
    name="Numba - no LUT",
    pen=pen_numba_yesLevels_noLUT
)

yesLUTnumba = floatPlot.plot(
    x=x,
    y=floatData["numba_yesLevels_uint16LUT"],
    name="(with LUT)",
    pen=pen_numba_yesLevels_uint16LUT
)

noLUTnumpy = floatPlot.plot(
    x=x,
    y=floatData["numpy_yesLevels_noLUT"],
    name="NumPy - no LUT",
    pen=pen_numpy_yesLevels_noLUT
)

yesLUTnumpy = floatPlot.plot(
    x=x,
    y=floatData["numpy_yesLevels_uint16LUT"],
    name="(with LUT)",
    pen=pen_numpy_yesLevels_uint16LUT
)


floatPlot.setLogMode(x=True, y=True)

cupyFillBetween = pg.FillBetweenItem(noLUTcupy, yesLUTcupy, brush=fillColor)
floatPlot.addItem(cupyFillBetween)

numpyFillBetween = pg.FillBetweenItem(noLUTnumpy, yesLUTnumpy, brush=fillColor)
floatPlot.addItem(numpyFillBetween)

numbaFillBetween = pg.FillBetweenItem(noLUTnumba, yesLUTnumba, brush=fillColor)
floatPlot.addItem(numbaFillBetween)

floatPlot.showGrid(x=True, y=True)
floatPlot.setLabel("left", "Image Update Duration (ms)")
floatPlot.setLabel("bottom", "Pixels in image")

uint16Plot.setTitle("16-bit integer performance", size=titleFontSize)
uint16Plot.axes["bottom"]["item"].enableAutoSIPrefix(False)

uint16Legend = uint16Plot.addLegend(brush="w", pen="k", verSpacing=-10, offset=(-5,-2), size=(10,10))

uint16Legend.setLabelTextSize(legendFontSize)
noLUTcupy = uint16Plot.plot(
    x=x,
    y=uint16Data["cupy_yesLevels_noLUT"],
    name="CuPy - no LUT",
    pen=pen_cupy_yesLevels_noLUT
)

yesLUTcupy = uint16Plot.plot(
    x=x,
    y=uint16Data["cupy_yesLevels_uint16LUT"],
    name="(with LUT)",
    pen=pen_cupy_yesLevels_uint16LUT
)

noLUTnumba = uint16Plot.plot(
    x=x,
    y=uint16Data["numba_yesLevels_noLUT"],
    name="Numba - no LUT",
    pen=pen_numba_yesLevels_noLUT
)

yesLUTnumba = uint16Plot.plot(
    x=x,
    y=uint16Data["numba_yesLevels_uint16LUT"],
    name="(with LUT)",
    pen=pen_numba_yesLevels_uint16LUT
)

noLUTnumpy = uint16Plot.plot(
    x=x,
    y=uint16Data["numpy_yesLevels_noLUT"],
    name="NumPy - no LUT",
    pen=pen_numpy_yesLevels_noLUT
)

yesLUTnumpy = uint16Plot.plot(
    x=x,
    y=uint16Data["numpy_yesLevels_uint16LUT"],
    name="(with LUT)",
    pen=pen_numpy_yesLevels_uint16LUT
)

# # special case
# noLUT_noLevels_numpy = uint16Plot.plot(
#     x=x,
#     y=uint16Data["numpy_noLevels_noLUT"],
#     name="NumPy - no LUT - no Levels",
#     pen=pen_blazing
# )

uint16Plot.setLogMode(x=True, y=True)

cupyFillBetween = pg.FillBetweenItem(noLUTcupy, yesLUTcupy, brush=fillColor)
uint16Plot.addItem(cupyFillBetween)

numpyFillBetween = pg.FillBetweenItem(noLUTnumpy, yesLUTnumpy, brush=fillColor)
uint16Plot.addItem(numpyFillBetween)

numbaFillBetween = pg.FillBetweenItem(noLUTnumba, yesLUTnumba, brush=fillColor)
uint16Plot.addItem(numbaFillBetween)

uint16Plot.showGrid(x=True, y=True)
uint16Plot.setLabel("bottom", "Pixels in image")

for plot in (floatPlot, uint16Plot):
    plot.setYRange( np.log10(0.09), np.log10(50), padding=0 )
    plot.setXRange( np.log10(5e4), np.log10(2e7), padding=0 )

win.resize(600, 500)
win.show()



if __name__ == '__main__':
    app.exec()