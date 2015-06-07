import numpy as np
from MoneyModel import MoneyModel
from mesa.visualization.ModularVisualization import (ModularServer,
                                                    VisualizationElement)
from mesa.visualization.modules import ChartModule

class HistogramModule(VisualizationElement):

    package_includes = ["Chart.min.js"]
    local_includes = ["HistogramModule.js"]
    canvas_height = 200
    canvas_width = 500

    def __init__(self, bins, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins, canvas_width, canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        wealth_vals = [agent.wealth for agent in model.schedule.agents]
        hist = np.histogram(wealth_vals, bins=self.bins)[0]

        return [int(x) for x in hist]

chart_element = ChartModule([{"Label": "Gini", "Color": "Black"}], 
                            data_collector_name='dc')

histogram_element = HistogramModule(range(10), 200, 500)
server = ModularServer(MoneyModel, [histogram_element], "MoneyModel", 100)
#server = ModularServer(MoneyModel, [chart_element], "Money Model", 100)
server.launch()