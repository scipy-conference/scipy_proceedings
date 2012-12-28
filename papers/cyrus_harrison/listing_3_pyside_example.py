#
# file: listing_3_pyside_example.py
#
# Usage:
#  >visit -cli -pysideviewer
#  >>>Source("listing_3_pyside_example.py")
#

class IsosurfaceWindow(QWidget):
    def __init__(self):
        super(IsosurfaceWindow,self).__init__()
        self.__init_widgets()
        # Setup our example plot.
        OpenDatabase("noise.silo")
        AddPlot("Pseudocolor","hardyglobal")
        AddOperator("Isosurface")
        self.update_isovalue(1.0)
        DrawPlots()
    def __init_widgets(self):
        # Create Qt layouts and widgets.
        vlout = QVBoxLayout(self)
        glout = QGridLayout()
        self.title   = QLabel("Iso Contour Sweep Example")
        self.title.setFont(QFont("Arial", 20, bold=True))
        self.sweep   = QPushButton("Sweep")
        self.lbound  = QLineEdit("1.0")
        self.ubound  = QLineEdit("99.0")
        self.step    = QLineEdit("2.0")
        self.current = QLabel("Current % =")
        f = QFont("Arial",bold=True,italic=True)
        self.current.setFont(f)
        self.rwindow = pyside_support.GetRenderWindow(1)
        # Add title and main render winodw.
        vlout.addWidget(self.title)
        vlout.addWidget(self.rwindow,10)
        glout.addWidget(self.current,1,3)
        # Add sweep controls.
        glout.addWidget(QLabel("Lower %"),2,1)
        glout.addWidget(QLabel("Upper %"),2,2)
        glout.addWidget(QLabel("Step %"),2,3)
        glout.addWidget(self.lbound,3,1)
        glout.addWidget(self.ubound,3,2)
        glout.addWidget(self.step,3,3)
        glout.addWidget(self.sweep,4,3)
        vlout.addLayout(glout,1)
        self.sweep.clicked.connect(self.exe_sweep)
        self.resize(600,600)
    def update_isovalue(self,perc):
        # Change the % value used by
        # the isosurface operator.
        iatts = IsosurfaceAttributes()
        iatts.contourMethod = iatts.Percent 
        iatts.contourPercent = (perc)
        SetOperatorOptions(iatts)
        txt = "Current % = "  + "%0.2f" % perc
        self.current.setText(txt)
    def exe_sweep(self):
        # Sweep % value accoording to 
        # the GUI inputs.
        lbv  = float(self.lbound.text())
        ubv  = float(self.ubound.text())
        stpv = float(self.step.text())
        v = lbv
        while v < ubv:
            self.update_isovalue(v)
            v+=stpv

# Create and show our custom window.
main = IsosurfaceWindow()
main.show()
