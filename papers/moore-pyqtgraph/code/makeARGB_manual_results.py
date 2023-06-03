import numpy as np
import pyqtgraph as pg

import pandas as pd
from math import log10

data = pd.read_csv("code/makeARGB_manual_results.csv")


app = pg.mkQApp("Video Speed Test Results")
pg.setConfigOptions(antialias=True, background="w", foreground="k")

# infiniteLineColor = "#00ACC1"
penWidth = 3

colorSetName = "viridis"

colormap = pg.colormap.get(colorSetName, source="matplotlib")
colors = colormap.getLookupTable(start=0.2, stop=0.5, nPts=2, mode="byte")

# color_noCUDA_yesLUT = colors[0]
# color_yesCUDA_yesLUT = colors[1]
# color_noCUDA_noLUT = colors[2]
# color_yesCUDA_noLUT = colors[3]



pen_noCUDA_yesLUT = pg.mkPen(colors[0], style=pg.QtCore.Qt.DashLine, width=penWidth)
pen_yesCUDA_yesLUT = pg.mkPen(colors[0], width=penWidth)

pen_noCUDA_noLUT = pg.mkPen(colors[1], style=pg.QtCore.Qt.DashLine, width=penWidth)
pen_yesCUDA_noLUT = pg.mkPen(colors[1], width=penWidth)

infiniteLineColorSetName = "inferno"
infiniteLineColorMap = pg.colormap.get(infiniteLineColorSetName, source="matplotlib")
infiniteLineColors = infiniteLineColorMap.getLookupTable(start=0.4, stop=0.9, nPts=4, mode="byte")

def getSeries(df):
    series = {}
    series["yesCUDA_yesLUT"] = np.reciprocal(df[(df["CUDA"] == "Yes") & (df["Lookup Table"] == "Yes")]["FPS"].to_numpy()) *1_000
    series["noCUDA_yesLUT"] = np.reciprocal(df[(df["CUDA"] == "No") & (df["Lookup Table"] == "Yes")]["FPS"].to_numpy()) *1_000
    series["yesCUDA_noLUT"] = np.reciprocal(df[(df["CUDA"] == "Yes") & (df["Lookup Table"] == "No")]["FPS"].to_numpy()) *1_000
    series["noCUDA_noLUT"] = np.reciprocal(df[(df["CUDA"] == "No") & (df["Lookup Table"] == "No")]["FPS"].to_numpy()) *1_000
    return series


win = pg.GraphicsLayoutWidget()

x = np.square([256, 512, 1024, 2048, 3072, 4096])

uint8Data = getSeries(data.loc[data["Dtype"] == "uint8"])


# uint8Plot = win.addPlot()
# uint8Plot.setTitle("Image Update Timing for uint8 dtype")
# uint8Plot.axes["bottom"]["item"].enableAutoSIPrefix(False)

# uint8Plot.addLegend()
# uint8Plot.plot(
#     x=x,
#     y=uint8Data["yesCUDA_yesLUT"],
#     name="CUDA with LUT",
#     pen=pen_yesCUDA_yesLUT
# )

# uint8Plot.plot(
#     x=x,
#     y=uint8Data["yesCUDA_noLUT"],
#     name="CUDA without LUT",
#     pen=pen_yesCUDA_noLUT
# )

# uint8Plot.plot(
#     x=x,
#     y=uint8Data["noCUDA_yesLUT"],
#     name="No CUDA with LUT",
#     pen=pen_noCUDA_yesLUT
# )
# uint8Plot.plot(
#     x=x,
#     y=uint8Data["noCUDA_noLUT"],
#     name="No CUDA without LUT",
#     pen=pen_noCUDA_noLUT
# )

# uint8Plot.addLine(
#     y=log10(1000 / 120),  # having to log10 due to log mode bug
#     label="120 FPS",
#     pen={
#         "color": infiniteLineColors[0],
#         "width": penWidth
#     },
#     labelOpts={
#         "position":1.0,
#         "color": "#212121",
#         "anchors":[(0.0, 0.0), (1.0, 0.0)]
#     }
# )
# uint8Plot.showGrid(x=True, y=True)

# uint8Plot.addLine(
#     y=log10(1000 / 60),  # having to log10 due to log mode bug
#     label="60 FPS",
#     pen={
#         "color": infiniteLineColors[3],
#         "width": penWidth
#     },
#     labelOpts={
#         "position":1.0,
#         "color": "#212121",
#         "anchors":[(0.0, 0.0), (1.0, 0.0)]
#     }
# )

# uint8Plot.addLine(
#     y=log10(1000 / 30),  # having to log10 due to log mode bug
#     label="30 FPS",
#     pen={
#         "color": infiniteLineColors[2],
#         "width": penWidth
#     },
#     labelOpts={
#         "position":0.0,
#         "color": "#212121",
#         "anchors":[(0.0, 0.0), (1.0, 0.0)]
#     }
# )

# uint8Plot.addLine(
#     y=log10(1000 / 15),  # having to log10 due to log mode bug
#     label="15 FPS",
#     pen={
#         "color": infiniteLineColors[3],
#         "width": penWidth
#     },
#     labelOpts={
#         "position":0.0,
#         "color": "#212121",
#         "anchors":[(0.0, 0.0), (1.0, 0.0)]
#     }
# )


# uint8Plot.setLabel("left", "Time to Update Frame (ms)")
# uint8Plot.setLabel("bottom", "Pixels in Image")
# uint8Plot.setLogMode(x=True, y=True)

uint16Data = getSeries(data.loc[data["Dtype"] == "uint16"])
uint16Plot = win.addPlot()
uint16Plot.setTitle("Image Update Timing for uint16 dtype")
uint16Plot.axes["bottom"]["item"].enableAutoSIPrefix(False)
uint16Plot.addLegend(brush="w", pen="k")
uint16Plot.plot(
    x=x,
    y=uint16Data["yesCUDA_yesLUT"],
    name="CUDA with LUT",
    pen=pen_yesCUDA_yesLUT
)

uint16Plot.plot(
    x=x,
    y=uint16Data["yesCUDA_noLUT"],
    name="CUDA without LUT",
    pen=pen_yesCUDA_noLUT
)

uint16Plot.plot(
    x=x,
    y=uint16Data["noCUDA_yesLUT"],
    name="No CUDA with LUT",
    pen=pen_noCUDA_yesLUT
)
uint16Plot.plot(
    x=x,
    y=uint8Data["noCUDA_noLUT"],
    name="No CUDA without LUT",
    pen=pen_noCUDA_noLUT
)
uint16Plot.showGrid(x=True, y=True)

uint16Plot.addLine(
    y=log10(1000 / 60),  # having to log10 due to log mode bug
    label="60 FPS",
    pen={
        "color": infiniteLineColors[3],
        "width": penWidth
    },
    labelOpts={
        "position":1.0,
        "color": "#212121",
        "anchors":[(0.0, 0.0), (1.0, 0.0)]
    }
)

uint16Plot.setLabel("left", "Time to Update Frame (ms)")
uint16Plot.setLabel("bottom", "Pixels in Image")
uint16Plot.setLogMode(x=True, y=True)


win.show()

if __name__ == '__main__':
    pg.mkQApp().exec_()