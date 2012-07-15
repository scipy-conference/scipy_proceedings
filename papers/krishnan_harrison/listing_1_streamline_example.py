#
# file: listing_1_streamline_example.py
#
# Usage:
#  >visit -cli -s listing_1_streamline_example.py
#

# Open an example file
OpenDatabase("noise.silo")
# Create a plot of the scalar field 'hardyglobal'
AddPlot("Pseudocolor","hardyglobal")
# Slice the volume to show only three 
# external faces.
AddOperator("ThreeSlice")
tatts = ThreeSliceAttributes()
tatts.x = -10
tatts.y = -10
tatts.z = -10
SetOperatorOptions(tatts)
DrawPlots()
# Find the maximum value of the field 'hardyglobal'
Query("Maximum")
val = GetQueryOutputValue()
print "Max value of 'hardyglobal' = ", val 
# Create a streamline plot that follows 
# the gradient of 'hardyglobal'
DefineVectorExpression("g","gradient(hardyglobal)")
AddPlot("Streamline","g")
satts = StreamlineAttributes()
satts.sourceType = satts.SpecifiedBox
satts.sampleDensity0 = 7
satts.sampleDensity1 = 7    
satts.sampleDensity2 = 7
satts.coloringMethod = satts.ColorBySeedPointID
SetPlotOptions(satts)
DrawPlots()