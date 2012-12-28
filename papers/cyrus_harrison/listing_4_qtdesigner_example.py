#
# file: listing_4_qtdesigner_example.py
#
# Usage:
#  >visit -cli -pyside -s listing_4_qtdesigner_example.py
#
#

from PySide.QtUiTools import *
# example slot
def on_my_button_click():
    print "myButton was clicked"

# Load a UI file created with QtDesigner
loader = QUiLoader()
uifile = QFile("custom_widget.ui")
uifile.open(QFile.ReadOnly)
main = loader.load(uifile)
# Use a string name to locate 
# objects from Qt UI file.
button = main.findChild(QPushButton, "myButton")
# After loading the UI, we want to 
# connect Qt slots to Python functions
button.clicked.connect(on_my_button_click)
main.show()