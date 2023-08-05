# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with 
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

app = pg.mkQApp("Plotting Example")

penWidth = 3
pg.setConfigOptions(antialias=True, background="w", foreground="k")

win = pg.GraphicsLayoutWidget(
    show=True,
)
win.resize(1000, 600)
win.setWindowTitle("PyQtGraph Example: Plotting")


p1 = win.addPlot(
    y=np.random.normal(size=100), pen={"color": "#757575", "width": penWidth}
)
p2 = win.addPlot()
p2.plot(
    np.random.normal(size=100),
    pen={"color": "#2196F3", "width": penWidth},
    name="Red curve",
)
p2.plot(
    np.random.normal(size=110) + 5,
    pen={"color": "#8D6E63", "width": penWidth},
    name="Green curve",
)
p2.plot(
    np.random.normal(size=120) + 10,
    pen={"color": "#F44336", "width": penWidth},
    name="Blue curve",
)

p3 = win.addPlot()
p3.plot(
    np.random.normal(size=100),
    pen={"color": "#757575", "width": penWidth},
    symbolBrush="#FFABAB",
    symbolPen="k",
)


win.nextRow()

p4 = win.addPlot()
x = np.cos(np.linspace(0, 2 * np.pi, 1000))
y = np.sin(np.linspace(0, 4 * np.pi, 1000))
p4.plot(x, y, pen={"color": "#424242", "width": penWidth})
p4.showGrid(x=True, y=True)

p5 = win.addPlot()
x = np.random.normal(size=1000) * 1e-5
y = x * 1000 + 0.005 * np.random.normal(size=1000)
y -= y.min() - 1.0
mask = x > 1e-15
x = x[mask]
y = y[mask]
p5.plot(
    x, y, pen=None, symbol="t", symbolPen=None, symbolSize=10, symbolBrush="#7E57C232"
)
# p5.setLabel('left', "Y Axis", units='A')
# p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)

p6 = win.addPlot()

## Create data

# To enhance the non-grid meshing, we randomize the polygon vertices per and
# certain amount
randomness = 5

# x and y being the vertices of the polygons, they share the same shape
# However the shape can be different in both dimension
xn = 50  # nb points along x
yn = 40  # nb points along y


x = (
    np.repeat(np.arange(1, xn + 1), yn).reshape(xn, yn)
    + np.random.random((xn, yn)) * randomness
)
y = (
    np.tile(np.arange(1, yn + 1), xn).reshape(xn, yn)
    + np.random.random((xn, yn)) * randomness
)
x.sort(axis=0)
y.sort(axis=0)


# z being the color of the polygons its shape must be decreased by one in each dimension
z = np.exp(-((x * xn) ** 2) / 1000)[:-1, :-1]

## Create autoscaling image item
edgecolors = None
antialiasing = False
cmap = pg.colormap.get("viridis")
levels = (-2, 2)  # Will be overwritten unless enableAutoLevels is set to False
# edgecolors = {'color':'w', 'width':2} # May be uncommented to see edgecolor effect
# antialiasing = True # May be uncommented to see antialiasing effect
# cmap         = pg.colormap.get('plasma') # May be uncommented to see a different colormap than the default 'viridis'
pcmi_auto = pg.PColorMeshItem(
    edgecolors=edgecolors,
    antialiasing=antialiasing,
    colorMap=cmap,
    levels=levels,
    enableAutoLevels=True,
)
p6.addItem(pcmi_auto)

# Wave parameters
wave_amplitude = 3
wave_speed = 0.3
wave_length = 10
color_speed = 0.3
color_noise_freq = 0.05

# display info in top-right corner
miny = np.min(y) - wave_amplitude
maxy = np.max(y) + wave_amplitude
p6.setYRange(miny, maxy)

i = 100
## Display the new data set
color_noise = np.sin(i * 2 * np.pi * color_noise_freq)
new_x = x
new_y = y + wave_amplitude * np.cos(x / wave_length + i)
new_z = (
    np.exp(-((x - np.cos(i * color_speed) * xn) ** 2) / 1000)[:-1, :-1] + color_noise
)
pcmi_auto.setData(new_x, new_y, new_z)
win.nextRow()

p7 = win.addPlot()
y = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(size=1000, scale=0.1)
p7.plot(y, fillLevel=-0.3, brush="#C8E6C9", pen={"color": "#757575", "width": penWidth})
p7.showAxis("bottom", False)

x2 = np.linspace(-100, 100, 1000)
data2 = np.sin(x2) / x2
p8 = win.addPlot()
p8.plot(data2, pen={"color": "#424242", "width": penWidth})
lr = pg.LinearRegionItem([400, 700], brush="#FFF9C4", pen="#1A237E")
lr.setZValue(-10)
p8.addItem(lr)

p9 = win.addPlot()
image_data = np.zeros((512, 512))
xx, yy = np.mgrid[:512, :512]
image_data = np.sin(xx / 50) * np.cos(yy / 50) * 20 + 20

p9ii = pg.ImageItem(image_data)
p9ii.setLookupTable(pg.colormap.get("inferno").getLookupTable())
p9.addItem(p9ii)
if __name__ == "__main__":
    pg.exec()
