import pyqtgraph as pg
from math import log10
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np

app = pg.mkQApp("MultiPlot Benchmark Results")
pg.setConfigOptions(antialias=True, background="w", foreground="k")

penWidth = 3
symbolSize = 7

# oneLinePenColor = "#b2df8a"
oneLinePenColor = "#C7472E"
# tenLinePenColor = "#a6cee3"
tenLinePenColor = "#B65FD3"
# hundredLinePenColor = "#DCD0FF"
hundredLinePenColor = "#41A7DC"
infiniteLineColor = "#FFB74D"

legendFontSize = "12pt"
titleFontSize = "14pt"

oneLineSymbol = "o"
tenLineSymbol = "t1"
hundredLineSymbol = "s"

# collected data
x = np.array([100, 1000, 10000, 100000, 1000000, 10000000], dtype=float)
oneLineData = np.array([2.33, 3.49, 9.08, 52.92, 490.53, 5449.26], dtype=float)
tenLineData = np.array([7.34, 9.96, 31.03, 240.66, 2441.94], dtype=float)
hundredLineData = np.array([34.544, 51.49, 224.36, 1909.70], dtype=float)

oneLineSubSampleData = np.array([2.26, 2.9, 6.81, 43.38, 383.15, 4042.9])
tenLineSubSampleData = np.array([5.17, 6.76, 26.49, 212.87, 2085.96])
hundredLineSubSampleData = np.array([23.16, 37.46, 209.97, 1811.96])

win = pg.GraphicsLayoutWidget(size=(750, 550))

pointsPerCurve = win.addPlot(row=0, col=0)
updateTimePerPoint = win.addPlot(row=0, col=1)
win.ci.layout.setColumnFixedWidth(0, 325)
win.ci.layout.setSpacing(25)


pointsPerCurve.axes["bottom"]["item"].enableAutoSIPrefix(False)

legend = pointsPerCurve.addLegend(offset=(5, 5), brush="w", pen="k")
legend.setLabelTextSize(legendFontSize)
pointsPerCurve.plot(
    x=x,
    y=oneLineData,
    name="1 line",
    pen={
        "color":oneLinePenColor,
        "width": penWidth,
    },
    symbolBrush=oneLinePenColor,
    symbolPen="k",
    symbolSize=symbolSize,
    symbol=oneLineSymbol
)
pointsPerCurve.plot(
    x=x,
    y=oneLineSubSampleData,
    name="(down-sampled)",
    pen={
        "color":oneLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.DashLine
    },
)

pointsPerCurve.plot(
    x=x[:-1],
    y=tenLineData,
    name="10 lines",
    pen={
        "color": tenLinePenColor,
        "width": penWidth
    },
    symbolBrush=tenLinePenColor,
    symbolSize=symbolSize,
    symbol=tenLineSymbol,
    symbolPen="k"
)

pointsPerCurve.plot(
    x=x[:-1],
    y=tenLineSubSampleData,
    name="(down-sampled)",
    pen={
        "color":tenLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.DashLine
    },
)

pointsPerCurve.plot(
    x=x[:-2],
    y=hundredLineData,
    name="100 lines",
    pen={
        "color": hundredLinePenColor,
        "width": penWidth
    },
    symbolBrush=hundredLinePenColor,
    symbolSize=symbolSize,
    symbol=hundredLineSymbol,
    symbolPen="k"
)

pointsPerCurve.plot(
    x=x[:-2],
    y=hundredLineSubSampleData,
    name="(down-sampled)",
    pen={
        "color":hundredLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.DashLine
    },
)

pointsPerCurve.setLabel("left", "Time to Update Frame (ms)")
pointsPerCurve.setLabel("bottom", "Points per Curve")
pointsPerCurve.addLine(
    y=log10(1000 / 60),  # having to log10 due to log mode bug
    label="60 FPS",
    pen={
        "color": infiniteLineColor,
        "width": penWidth
    },
    labelOpts={
        "position":1.0,
        "color": "#212121",
        "anchors":[(0.0, 0.0), (1.0, 0.0)]
    }
)
pointsPerCurve.addLine(
    y=log10(1000 / 10),  # having to log10 due to log mode bug
    label="10 FPS",
    pen={
        "color": infiniteLineColor,
        "width": penWidth
    },
    labelOpts={
        "position":1.0,
        "color":"#212121",
        "anchors":[(0.0, 0.0), (1.0, 0.0)]
        }
    )

pointsPerCurve.setLogMode(x=True, y=True)

#####

legend = updateTimePerPoint.addLegend(offset=(-10, 5), brush="w", pen="k")
legend.setLabelTextSize(legendFontSize)

updateTimePerPoint.setLabel("left", "Update Time per Point (µs)")
updateTimePerPoint.setLabel("bottom", "Total Points")
updateTimePerPoint.axes["bottom"]["item"].enableAutoSIPrefix(False)

totalPointsOneLine = x.copy()
totalPointsTenLine = x[:-1] * 10
totalPointsHundredLine = x[:-2] * 100 

perPointUpdateDurationOneLine = 1_000 * oneLineData / totalPointsOneLine
perPointUpdateDurationsTenLines = 1_000 * tenLineData / totalPointsTenLine
perPointUpdateDurationsHundredLines = 1_000 * hundredLineData / totalPointsHundredLine

perPointUpdateSubSampleDurationOneLine = 1_000 * oneLineSubSampleData / totalPointsOneLine
perPointUpdateSubSampleDurationsTenLines = 1_000 * tenLineSubSampleData / totalPointsTenLine
perPointUpdateSubSampleDurationsHundredLines = 1_000 * hundredLineSubSampleData / totalPointsHundredLine



updateTimePerPoint.plot(
    x=totalPointsHundredLine,
    y=perPointUpdateDurationsHundredLines,
    pen={
        "color": hundredLinePenColor,
        "width": penWidth
    },
    symbol=hundredLineSymbol,
    symbolPen="k",
    symbolSize=symbolSize,
    symbolBrush=hundredLinePenColor,
    name="100 lines"
)

updateTimePerPoint.plot(
    x=totalPointsHundredLine,
    y=perPointUpdateSubSampleDurationsHundredLines,
    pen={
        "color": hundredLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.DashLine
    },
    name="(down-sampled)",
)

updateTimePerPoint.plot(
    x=totalPointsOneLine,
    y=perPointUpdateDurationOneLine,
    pen={
        "color": oneLinePenColor,
        "width": penWidth
    },
    symbol=oneLineSymbol,
    symbolPen="k",
    symbolSize=symbolSize,
    symbolBrush=oneLinePenColor,
    name="1 line"
)

updateTimePerPoint.plot(
    x=totalPointsOneLine,
    y=perPointUpdateSubSampleDurationOneLine,
    pen={
        "color": oneLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.DashLine
    },
    name="(down-sampled)",
)

updateTimePerPoint.plot(
    x=totalPointsTenLine,
    y=perPointUpdateDurationsTenLines,
    pen={
        "color": tenLinePenColor,
        "width": penWidth
    },
    symbol=tenLineSymbol,
    symbolPen="k",
    symbolSize=symbolSize,
    symbolBrush=tenLinePenColor,
    name="10 lines"
)

updateTimePerPoint.plot(
    x=totalPointsTenLine,
    y=perPointUpdateSubSampleDurationsTenLines,
    pen={
        "color": tenLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.DashLine
    },
    name="(down-sampled)",
)


updateTimePerPoint.setLogMode(x=True, y=True)
updateTimePerPoint.addLine(
    y=log10(200 / 1_000),
    pen={
        "color": "k",
        "width": penWidth - 2,
        "style": QtCore.Qt.DashLine
    },
    label="200 ns",
    labelOpts={
        "position":0.0,
        "color":"#212121",
        "anchors": [(0.0, 0.0), (0.0, 1.0)]
    }
)

updateTimePerPoint.showGrid(x=True, y=True)

win.show()

if __name__ == '__main__':
    pg.mkQApp().exec_()
