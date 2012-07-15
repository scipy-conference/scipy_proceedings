#
# file: listing_6_invoke_cell_average_example.py
#
# Usage:
#  >visit -cli -s listing_6_invoke_cell_average_example.py
#


# Open an example data set.
OpenDatabase("multi_rect3d.silo")
# Create a plot to query
AddPlot("Pseudocolor","d")
DrawPlots()
# Execute the Python query
PythonQuery(file="listing_5_cell_average.vpq",
            vars=["default"])