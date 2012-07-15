#
# file: listing_2_import_example.py
#
# Usage:
# See detailed instructions at:
#  http://visitusers.org/index.php?title=Python_Module_Support
# 

import sys
import os
from os.path import join as pjoin
vpath = "path/to/visit/<ver>/<arch>/"
# or for an OSX bundle version
# "path/to/VisIt.app/Contents/Resources/<ver>/<arch>"
vpath = pjoin(vpath,"lib","site-packages")
sys.path.insert(0,vpath)
import visit
visit.Launch()
# use the interface
visit.OpenDatabase("noise.silo")
visit.AddPlot("Pseudocolor","hardyglobal")