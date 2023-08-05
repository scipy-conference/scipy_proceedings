import pyqtgraph as pg
from math import log10
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np

from PyQt5.QtGui import QFont


app = pg.mkQApp("MultiPlot Benchmark Results")
pg.setConfigOptions(antialias=True, background="w", foreground="k")

penWidth = 3
symbolSize = 7

legendFontSize = "10pt"
titleFontSize = "12pt"
axisFontSize = "10pt"
fontL = QFont("Arial")
fontL.setPointSize(14)

fontM = QFont("Arial")
fontM.setPointSize(12)

# oneLinePenColor = "#b2df8a"
oneLinePenColor = "#C7472E"
# tenLinePenColor = "#a6cee3"
tenLinePenColor = "#B65FD3"
# hundredLinePenColor = "#DCD0FF"
hundredLinePenColor = "#41A7DC"
infiniteLineColor = "#FFB74D"

# legendFontSize = "12pt"
# titleFontSize = "14pt"

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
pointsPerCurve.setFixedWidth(280)

updateTimePerPoint = win.addPlot(row=0, col=1)
updateTimePerPoint.setFixedWidth(280)

# win.ci.layout.setColumnFixedWidth(0, 325)
# win.ci.layout.setSpacing(25)
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

legend = pointsPerCurve.addLegend(offset=(-5, -2), brush="w", pen="k", verSpacing=-10, size=(10,10))


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

# pointsPerCurve.setLabel("left", "Time to Update Frame (ms)")
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

# legend.setLabelTextSize(legendFontSize)

# updateTimePerPoint.setLabel("left", "Update Time per Point (µs)")
updateTimePerPoint.setLabel("bottom", "Total points")
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
line = updateTimePerPoint.addLine(
    y=log10(200 / 1_000),
    pen={
        "color": "k",
        "width": penWidth - 1.5,
        "style": QtCore.Qt.DashLine
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

updateTimePerPoint.setYRange( np.log10(0.15), np.log10(30), padding=0 )
pointsPerCurve.setYRange( np.log10(1), np.log10(8e3), padding=0 )

for plot in (pointsPerCurve, updateTimePerPoint):
    plot.setXRange( np.log10(50.), np.log10(2e7), padding=0 )

win.show()

if __name__ == '__main__':
    pg.mkQApp().exec_()
