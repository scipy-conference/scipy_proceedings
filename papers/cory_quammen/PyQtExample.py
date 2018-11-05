import sys
import vtk
from PyQt4 import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor \
    import QVTKRenderWindowInteractor

class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent = None):
        QtGui.QMainWindow.__init__(self, parent)

        self.frame = QtGui.QFrame()

        layout = QtGui.QVBoxLayout()
        self.vtkWidget = \
            QVTKRenderWindowInteractor(self.frame)
        layout.addWidget(self.vtkWidget)

        self.renderer = vtk.vtkRenderer()
        rw = self.vtkWidget.GetRenderWindow()
        rw.AddRenderer(self.renderer)
        self.interactor = rw.GetInteractor()

        cylinder = vtk.vtkCylinderSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection( \
            cylinder.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        self.frame.setLayout(layout)
        self.setCentralWidget(self.frame)

        self.show()
        self.interactor.Initialize()
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
