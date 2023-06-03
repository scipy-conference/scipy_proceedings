# -*- coding: utf-8 -*-
"""
This example demonstrates the ability to link the axes of views together
Views can be linked manually using the context menu, but only if they are given 
names.
"""

import sys
import time
# from pyqtgraph.Qt import QtGui, QtCore
from PyQt5 import QtGui, QtCore
import PyQt5.QtWidgets as QtWidgets

import numpy as np
import pyqtgraph as pg

# app = pg.mkQApp("Linked Views Example")
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

GOOD_Y1 = 10
STD_Y1 = 0.2
GOOD_Y2 = 0.5
STD_Y2 = 0.3

class Generator(object):
    def __init__(self):
        self.data_fps = 100
        self.tstep = 1/self.data_fps
        self.len = 5_000
        self.xdata  = np.arange(self.len) * self.tstep
        self.ydata1 = np.random.random( size=len( self.xdata ) )
        self.ydata2 = np.random.random( size=len( self.xdata ) )
        self.frame_time = None
        self.last_time = time.time()
        self.t_fail = self.last_time + (1+ 5*np.random.random() )
    def get_data(self):
        new_time = time.time()
        time_passed = new_time - self.last_time
        self.last_time = new_time
        
        if self.frame_time is None:
            self.frame_time = time_passed
        else:
            s = 0.02 # 50 frame average
            self.frame_time = self.frame_time * (1-s) + time_passed * s

        new_samples = int( self.data_fps * time_passed )
        if new_samples > 0:
            # if new_samples > 2: print('.')
            new_x  = self.xdata[-1] + self.tstep * (np.arange(1, new_samples+1) )

            if self.t_fail > new_time: # simulate good data
                new_y1 = np.random.normal( size=new_samples ) * STD_Y1 + GOOD_Y1
                new_y2 = np.random.normal( size=new_samples ) * STD_Y2 + GOOD_Y2
            else:
                time_since_failure = new_time - self.t_fail
                act_y1 = GOOD_Y1 * 1 / (1+ 75*time_since_failure)
                if act_y1 < 0: act_y = 0
                new_y1 = np.random.normal( size=new_samples ) * STD_Y1 + act_y1
                act_std_y2 = STD_Y1 / (act_y1 / GOOD_Y1 )
                new_y2 = np.random.normal( size=new_samples ) * act_std_y2 + GOOD_Y2
                
                if time_since_failure > 0.15: 
                    self.t_fail = self.last_time + (5+ 25*np.random.random() ) # schedule next failule :)
                    # self.t_fail = new_time + 30
            # roll data
            self.xdata [:-new_samples] = self.xdata [new_samples:]
            self.ydata1[:-new_samples] = self.ydata1[new_samples:]
            self.ydata2[:-new_samples] = self.ydata2[new_samples:]
            
            self.xdata [-new_samples:] = new_x
            self.ydata1[-new_samples:] = new_y1
            self.ydata2[-new_samples:] = new_y2
        return(self.xdata, self.ydata1, self.ydata2, self.frame_time)
        
class Demo( pg.GraphicsLayoutWidget ):
    def __init__(self):
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        super().__init__()
        self.gen = Generator()
        # self.setTitle('Measurement monitor')
        # win = pg.GraphicsLayoutWidget(show=True, title="pyqtgraph example: Linked Views")
        self.resize(350,350)
        # pen1 = QtGui.QColor('#A64D21') # orange
        # pen2 = QtGui.QColor('#BFB226') # yellow
        pen1 = QtGui.QColor('#572491') # orange
        pen2 = QtGui.QColor('#1c4d91') # yellow
        self.p1 = pg.PlotCurveItem( x=[], y=[], pen=pen1, title="operating condition")
        self.p2 = pg.PlotCurveItem( x=[], y=[], pen=pen2, title="measured signal")

        self.plot1 = self.addPlot(name='Plot1')
        self.plot1.setLabel('left', "monitored condition")
        self.nextRow()
        plot2 = self.addPlot(name='Plot2')
        plot2.setXLink('Plot1')  ## test linking by name
        plot2.setLabel('bottom', "operating time (s)")
        plot2.setLabel('left', "measurement")

        for plot in [self.plot1, plot2]:
            ax = plot.getAxis('left')
            ax.setStyle(tickTextWidth=20)
            ax.show()
            ax = plot.getAxis('right')
            ax.setStyle(tickTextWidth=20)
            ax.show()
            ax = plot.getAxis('top')
            ax.setStyle(showValues=False)
            ax.show()

        self.plot1.addItem( self.p1 )
        plot2.addItem( self.p2 )

        QtGui.QApplication.processEvents()
        self.timer = QtCore.QTimer( singleShot=False )
        self.timer.timeout.connect(self.update)
        self.timer.start(20) # 20 fps
        self.update()
        
    def update(self):
        (xdata, ydata1, ydata2, frame_time) = self.gen.get_data()
        self.p1.setData(xdata, ydata1)
        self.p2.setData(xdata, ydata2)
        self.plot1.setLabel('top', 'update rate: {:.1f} FPS'.format(1/frame_time) )
        # QtGui.QApplication.processEvents()

# if __name__ == '__main__':
#     pg.mkQApp().exec_()
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = Demo()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
