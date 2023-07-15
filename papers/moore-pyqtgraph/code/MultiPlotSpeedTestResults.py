import pyqtgraph as pg
from math import log10
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np

app = pg.mkQApp("MultiPlot Benchmark Results")
pg.setConfigOptions(antialias=True, background="w", foreground="k")

penWidth = 3
symbolSize = 7

legendFontSize = "10pt"
titleFontSize = "12pt"
axisFontSize = "10pt"
fontL = QtGui.QFont("Arial")
fontL.setPointSize(14)

fontM = QtGui.QFont("Arial")
fontM.setPointSize(12)

# oneLinePenColor = "#b2df8a"
oneLinePenColor = "#C7472E"
# tenLinePenColor = "#a6cee3"
tenLinePenColor = "#B65FD3"
# hundredLinePenColor = "#DCD0FF"
hundredLinePenColor = "#41A7DC"
infiniteLineColor = "#FFB74D"


oneLineSymbol = "o"
tenLineSymbol = "t1"
hundredLineSymbol = "s"

# collected data
x = np.array([100, 1_000, 10_000, 100_000, 1_000_000], dtype=float)

oneLineData = np.array([    0.6798570941,   0.9951128181,   4.074828883, 33.93462454, 1211.236254])
tenLineData = np.array([    2.619823496,    3.079782973,    11.591055,   126.2503826, 2478.585986])
hundredLineData = np.array([13.77128709,    20.49453436,    79.33722078, 453.3641617, 13078.61032])


oneLineSubSampleData = np.array([0.5991630017, 0.9227964749, 3.668691927, 5.755929296, 10.74499962])
tenLineSubSampleData = np.array([2.58146673, 2.994403873, 11.67468732, 16.68522375, 55.09732074])
hundredLineSubSampleData = np.array([14.0038153, 21.07908316, 80.17760808, 106.37557302788, 578.9639407])

win = pg.GraphicsLayoutWidget(size=(750, 550))

pointsPerCurve = win.addPlot(row=0, col=0)
pointsPerCurve.setFixedWidth(280)

updateTimePerPoint = win.addPlot(row=0, col=1)
updateTimePerPoint.setFixedWidth(280)

for plot in (pointsPerCurve, updateTimePerPoint):
    plot.disableAutoRange()
    plot.titleLabel.setFont( fontL )
    plot.axes["bottom"]["item"].enableAutoSIPrefix(False)
    for loc in ('left','right','top','bottom'):
        ax = plot.getAxis(loc)
        # ax.setFont( fontL )
        ax.setStyle( tickFont=fontM )
        if loc in('right', 'top'): ax.setStyle( showValues=False )
        ax.label.setFont( fontM )
        ax.show()

pointsPerCurve.setTitle("Time to update frame  (ms)")
updateTimePerPoint.setTitle("Update time per point (µs)")

pointsPerCurve.axes["bottom"]["item"].enableAutoSIPrefix(False)

legend = pointsPerCurve.addLegend(offset=(5, 2), brush="w", pen="k", verSpacing=-10, size=(10,10))


legend.setLabelTextSize(legendFontSize)
pointsPerCurve.plot(
    x=x,
    y=oneLineData,
    name="1 curve",
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
        "style": QtCore.Qt.PenStyle.DashLine
    },
)

pointsPerCurve.plot(
    x=x,
    y=tenLineData,
    name="10 curves",
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
    x=x,
    y=tenLineSubSampleData,
    name="(down-sampled)",
    pen={
        "color":tenLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.PenStyle.DashLine
    },
)

pointsPerCurve.plot(
    x=x,
    y=hundredLineData,
    name="100 curves",
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
    x=x,
    y=hundredLineSubSampleData,
    name="(down-sampled)",
    pen={
        "color":hundredLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.PenStyle.DashLine
    },
)

pointsPerCurve.setLabel("bottom", "Points per curve")
line60 = pointsPerCurve.addLine(
    y=log10(1000 / 60.),  # having to log10 due to log mode bug
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
line10 = pointsPerCurve.addLine(
    y=log10(1000 / 10.),  # having to log10 due to log mode bug
    label="10 FPS",
    pen={
        "color": infiniteLineColor,
        "width": penWidth
    },
    labelOpts={
        "position":1.0,
        "color":"#212121",
        "anchors":[(1.0, 0.0), (1.0, 0.0)]
        }
    )

line10.label.setFont( fontM )
line60.label.setFont( fontM )



pointsPerCurve.setLogMode(x=True, y=True)

#####

# legend = updateTimePerPoint.addLegend(offset=(-10, 5), brush="w", pen="k")
legend = updateTimePerPoint.addLegend(brush="w", pen="k", verSpacing=-10, offset=(-5,5), size=(10,10))

# updateTimePerPoint.setLabel("left", "Update Time per Point (µs)")
updateTimePerPoint.setLabel("bottom", "Total points")
updateTimePerPoint.axes["bottom"]["item"].enableAutoSIPrefix(False)

totalPointsOneLine = x.copy()
totalPointsTenLine = x * 10
totalPointsHundredLine = x * 100

perPointUpdateDurationOneLine = 1_000 * oneLineData / totalPointsOneLine
perPointUpdateDurationsTenLines = 1_000 * tenLineData / totalPointsTenLine
perPointUpdateDurationsHundredLines = 1_000 * hundredLineData / totalPointsHundredLine

perPointUpdateSubSampleDurationOneLine = 1_000 * oneLineSubSampleData / totalPointsOneLine
perPointUpdateSubSampleDurationsTenLines = 1_000 * tenLineSubSampleData / totalPointsTenLine
perPointUpdateSubSampleDurationsHundredLines = 1_000 * hundredLineSubSampleData / totalPointsHundredLine

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
    name="1 curve"
)

updateTimePerPoint.plot(
    x=totalPointsOneLine,
    y=perPointUpdateSubSampleDurationOneLine,
    pen={
        "color": oneLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.PenStyle.DashLine
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
    name="10 curves"
)

updateTimePerPoint.plot(
    x=totalPointsTenLine,
    y=perPointUpdateSubSampleDurationsTenLines,
    pen={
        "color": tenLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.PenStyle.DashLine
    },
    name="(down-sampled)",
)

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
    name="100 curves"
)

updateTimePerPoint.plot(
    x=totalPointsHundredLine,
    y=perPointUpdateSubSampleDurationsHundredLines,
    pen={
        "color": hundredLinePenColor,
        "width": penWidth - 1,
        "style": QtCore.Qt.PenStyle.DashLine
    },
    name="(down-sampled)",
)

updateTimePerPoint.setLogMode(x=True, y=True)
line = updateTimePerPoint.addLine(
    y=log10(200 / 1_000),
    pen={
        "color": "k",
        "width": penWidth - 1.5,
        "style": QtCore.Qt.PenStyle.DashLine
    },
    label="200 ns",
    labelOpts={
        "position":0.1,
        "color":"#212121",
        "anchors": [(0.0, 0.0), (0.0, 1.0)]
    }
)
line.label.setFont( fontM )

updateTimePerPoint.showGrid(x=True, y=True)

updateTimePerPoint.setYRange( np.log10(0.005), np.log10(10.1), padding=0 )
updateTimePerPoint.setXRange( np.log(5.), np.log10(1.2e8), padding=0)

pointsPerCurve.setYRange( np.log10(0.5), np.log10(1.5e4), padding=0 )
pointsPerCurve.setXRange( np.log10(50.), np.log10(2e7), padding=0)

win.resize(600, 500)
win.show()

if __name__ == '__main__':
    pg.exec()