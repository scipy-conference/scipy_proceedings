import numpy as np
import pyqtgraph as pg

import pandas as pd
from math import log10

from PyQt5.QtGui import QFont

data = pd.read_csv(
    "code/makeARGB_benchmark_results.csv",
    names=["size", "CuPy", "dtype", "levels", "LUT", "time"],
    converters={
        "CuPy": lambda x: x == "cupy",
        "levels": lambda x: x == "levels"
    },
    header=0
)

app = pg.mkQApp("Video Speed Test Results")
pg.setConfigOptions(antialias=True, background="w", foreground="k")

penWidth = 3
legendFontSize = "10pt"
titleFontSize = "12pt"
axisFontSize = "10pt"
fontL = QFont("Arial")
fontL.setPointSize(14)

fontM = QFont("Arial")
fontM.setPointSize(12)

gray = "#757575"
blue = "#41A7DC"
purple = "#B65FD3"
fillColor = "#00004040"

pen_noCUDA_yesLevels_noLUT = pg.mkPen(purple, style=pg.QtCore.Qt.DashLine, width=penWidth)
pen_yesCUDA_yesLevels_noLUT = pg.mkPen(purple, width=penWidth)

# pen_noCUDA_noLevels_noLUT = pg.mkPen(colors[1], style=pg.QtCore.Qt.DashLine, width=penWidth)
# pen_yesCUDA_noLevels_noLUT = pg.mkPen(colors[1], width=penWidth)

# pen_noCUDA_yesLevels_unint8LUT = pg.mkPen(colors[2], style=pg.QtCore.Qt.DashLine, width=penWidth)
# pen_yesCUDA_yesLevels_uint8LUT = pg.mkPen(colors[2], width=penWidth)

# pen_noCUDA_noLevels_uint8LUT = pg.mkPen(colors[3], style=pg.QtCore.Qt.DashLine, width=penWidth)
# pen_yesCUDA_noLevels_uint8LUT = pg.mkPen(colors[3], width=penWidth)


pen_noCUDA_yesLevels_uint16LUT = pg.mkPen(blue, style=pg.QtCore.Qt.DashLine, width=penWidth)
pen_yesCUDA_yesLevels_uint16LUT = pg.mkPen(blue, width=penWidth)

# pen_noCUDA_noLevels_uint16LUT = pg.mkPen(colors[5], style=pg.QtCore.Qt.DashLine, width=penWidth)
# pen_yesCUDA_noLevels_uint16LUT = pg.mkPen(colors[5], width=penWidth)


infiniteLineColorSetName = "inferno"
infiniteLineColorMap = pg.colormap.get(infiniteLineColorSetName, source="matplotlib")
infiniteLineColors = infiniteLineColorMap.getLookupTable(start=0.1, stop=0.9, nPts=4, mode="byte")

def getSeries(df):
    series = {}
    series["yesCUDA_yesLevels_noLUT"] = df[(df["CuPy"]) & (df["levels"]) & (df["LUT"] == "nolut")]["time"].to_numpy() * 1_000
    series["yesCUDA_yesLevels_uint8LUT"] = df[(df["CuPy"]) & (df["levels"]) & (df["LUT"] == "uint8lut")]["time"].to_numpy() * 1_000
    series["yesCUDA_yesLevels_uint16LUT"] = df[(df["CuPy"]) & (df["levels"]) & (df["LUT"] == "uint16lut")]["time"].to_numpy() * 1_000

    series["yesCUDA_noLevels_noLUT"] = df[(df["CuPy"]) & (df["levels"] == False) & (df["LUT"] == "nolut")]["time"].to_numpy() * 1_000
    series["yesCUDA_noLevels_uint8LUT"] = df[(df["CuPy"]) & (df["levels"] == False) & (df["LUT"] == "uint8lut")]["time"].to_numpy() * 1_000
    series["yesCUDA_noLevels_uint16LUT"] = df[(df["CuPy"]) & (df["levels"] == False) & (df["LUT"] == "uint16lut")]["time"].to_numpy() * 1_000

    series["noCUDA_yesLevels_noLUT"] = df[(df["CuPy"] == False) & (df["levels"]) & (df["LUT"] == "nolut")]["time"].to_numpy() * 1_000
    series["noCUDA_yesLevels_uint8LUT"] = df[(df["CuPy"] == False) & (df["levels"]) & (df["LUT"] == "uint8lut")]["time"].to_numpy() * 1_000
    series["noCUDA_yesLevels_uint16LUT"] = df[(df["CuPy"] == False) & (df["levels"]) & (df["LUT"] == "uint16lut")]["time"].to_numpy() * 1_000

    series["noCUDA_noLevels_noLUT"] = df[(df["CuPy"] == False) & (df["levels"] == False) & (df["LUT"] == "nolut")]["time"].to_numpy() * 1_000
    series["noCUDA_noLevels_uint8LUT"] = df[(df["CuPy"] == False) & (df["levels"] == False) & (df["LUT"] == "uint8lut")]["time"].to_numpy() * 1_000
    series["noCUDA_noLevels_uint16LUT"] = df[(df["CuPy"] == False) & (df["levels"] == False) & (df["LUT"] == "uint16lut")]["time"].to_numpy() * 1_000

    return series


win = pg.GraphicsLayoutWidget()

x = np.square([256, 512, 1024, 2048, 3072, 4096])

uint8Data = getSeries(data.loc[data["dtype"] == "uint8"])
uint16Data = getSeries(data.loc[data["dtype"] == "uint16"])
floatData = getSeries(data.loc[data["dtype"] == "float"])

floatPlot = win.addPlot(0,0, 1,1)
floatPlot.setFixedWidth(290)

uint16Plot = win.addPlot(0,2, 1,1)
uint16Plot.setFixedWidth(270)

print( win.ci.layout) 
win.ci.layout.setColumnFixedWidth(1,1)

for plot in (floatPlot, uint16Plot):
    plot.disableAutoRange()
    plot.setYRange( 0.1, 200, padding=0 )
    plot.titleLabel.setFont( fontL )
    plot.axes["bottom"]["item"].enableAutoSIPrefix(False)
    for loc in ('left','right','top','bottom'):
        ax = plot.getAxis(loc)
        # ax.setFont( fontL )
        ax.setStyle( tickFont=fontM )
        if loc in('right', 'top'): ax.setStyle( showValues=False )
        ax.label.setFont( fontM )
        ax.show()

floatPlot.setTitle("Float format performance", size=titleFontSize)

floatLegend = floatPlot.addLegend(brush="w", pen="k", verSpacing=-10, offset=(-5,-2), size=(10,10))
# floatLegend.setLabelTextSize(legendFontSize)
noLUTyesCUDA = floatPlot.plot(
    x=x,
    y=floatData["yesCUDA_yesLevels_noLUT"],
    name="no LUT - CUDA enabled",
    pen=pen_yesCUDA_yesLevels_noLUT
)

noLUTnoCUDA = floatPlot.plot(
    x=x,
    y=floatData["noCUDA_yesLevels_noLUT"],
    name="no LUT - NumPy only",
    pen=pen_noCUDA_yesLevels_noLUT
)

yesLUTyesCUDA = floatPlot.plot(
    x=x,
    y=floatData["yesCUDA_yesLevels_uint16LUT"],
    name="16-bit LUT - CUDA enabled",
    pen=pen_yesCUDA_yesLevels_uint16LUT
)

yesLUTnoCUDA = floatPlot.plot(
    x=x,
    y=floatData["noCUDA_yesLevels_uint16LUT"],
    name="16-bit LUT - NumPy Only",
    pen=pen_noCUDA_yesLevels_uint16LUT
)

floatPlot.setLogMode(x=True, y=True)

yesCUDAFillBetween = pg.FillBetweenItem(noLUTyesCUDA, yesLUTyesCUDA, brush=fillColor)
floatPlot.addItem(yesCUDAFillBetween)

noCUDAFillBetween = pg.FillBetweenItem(noLUTnoCUDA, yesLUTnoCUDA, brush=fillColor)
floatPlot.addItem(noCUDAFillBetween)

floatPlot.showGrid(x=True, y=True)
floatPlot.setLabel("left", "makeARGB run time (ms)")
floatPlot.setLabel("bottom", "Pixels in image")



uint16Plot.setTitle("16-bit integer performance", size=titleFontSize)
uint16Plot.axes["bottom"]["item"].enableAutoSIPrefix(False)

uint16Legend = uint16Plot.addLegend(brush="w", pen="k", verSpacing=-10, offset=(-5,-2), size=(10,10))

# uint16Legend.setLabelTextSize(legendFontSize)
noLUTyesCUDA = uint16Plot.plot(
    x=x,
    y=uint16Data["yesCUDA_yesLevels_noLUT"],
    name="no LUT - CUDA enabled",
    pen=pen_yesCUDA_yesLevels_noLUT
)

noLUTnoCUDA = uint16Plot.plot(
    x=x,
    y=uint16Data["noCUDA_yesLevels_noLUT"],
    name="no LUT - NumPy only",
    pen=pen_noCUDA_yesLevels_noLUT
)

yesLUTyesCUDA = uint16Plot.plot(
    x=x,
    y=uint16Data["yesCUDA_yesLevels_uint16LUT"],
    name="16-bit LUT - CUDA enabled",
    pen=pen_yesCUDA_yesLevels_uint16LUT
)

yesLUTnoCUDA = uint16Plot.plot(
    x=x,
    y=uint16Data["noCUDA_yesLevels_uint16LUT"],
    name="16-bit LUT - NumPy Only",
    pen=pen_noCUDA_yesLevels_uint16LUT
)

uint16Plot.setLogMode(x=True, y=True)

yesCUDAFillBetween = pg.FillBetweenItem(noLUTyesCUDA, yesLUTyesCUDA, brush=fillColor)
uint16Plot.addItem(yesCUDAFillBetween)

noCUDAFillBetween = pg.FillBetweenItem(noLUTnoCUDA, yesLUTnoCUDA, brush=fillColor)
uint16Plot.addItem(noCUDAFillBetween)

uint16Plot.showGrid(x=True, y=True)
# uint16Plot.setLabel("left", "makeARGB Runtime (ms)")  # do not repeat
uint16Plot.setLabel("bottom", "Pixels in image")

for plot in (floatPlot, uint16Plot):
    plot.setYRange( np.log10(0.05), np.log10(200), padding=0 )
    plot.setXRange( np.log10(5e4), np.log10(2e7), padding=0 )

win.resize(1000, 500)
win.show()

if __name__ == '__main__':
    pg.mkQApp().exec_()