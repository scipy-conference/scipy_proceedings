Extras
------
Figures may be displayed in the notebook using the legacy iplot methods
[ref], or by using the new ``FigureWidget`` subclass described section X.


Codegen:
Numpy doc type annotations to support completion in PyCharm. Constructor 
docstrings.


Diagrams
--------

### Plotly data model

```
{'data': [
            {'type': 'scatter', 'y': [1, 3, 2]},
            {'type': 'bar', 'y': [3, 2, 4]}
            ],
     'layout': {
        'xaxis': {'range': [-1, 5]}
     }}
```
+
Figure thumbnail

Caption: json structure representing a figure with two traces 


### Screenshot of multiple view of ipywidget
Image of slider in notebook output cell and side pane

Maybe roll this into another example later on


### ipywidgets architecture
```
Python          JavaScript

                        [view]
              Comm      /
    [model] -------  [model]
                        \
                        [view]
```

Caption: Architecture

## Figure API Example

This will be a full page figure with 7 code snippets, json representations, and 

```python
import plotly.graph_objs as go
fig = go.Figure(data=[
    go.Scatter(y=[2, 3, 1],
               marker=go.scatter.Marker(color='green'))])  # [A]

fig.to_plotly_json() == {
    'data': [{'y': [2, 3, 1], 'marker': {'color':'green'}}],
    'layout': {}}

fig.layout.xaxis.range = [-1, 5]  # [B]
fig.to_plotly_json() == {
    'data': [{'y': [2, 3, 1], 'marker': {'color':'green'}}],
    'layout': {'xaxis': {'range': [-1, 5]}}}

fig.add_bar(y=[3, 2, 4])  # [C]
fig.to_plotly_json() == {
    'data': [{'y': [2, 3, 1], 'marker': {'color':'green'}},
             {'y': [3, 2, 4]}],
    'layout': {'xaxis': {'range': [-1, 5]}}}

fig.data = [fig.data[1], fig.data[0]]  # [D]
fig.to_plotly_json()

fig.data = [fig.data[1]]  # [E]
fig.to_plotly_json()

with fig.batch_update():  # [F]
    fig.data[0].x = [-1, 0, 1]
    fig.layout.xaxis.range = [-2, 4] 

with fig.batch_animate(duration=2000):  # [G]
    fig.layout.xaxis.range = [-10, 10]
    fig.layout.yaxis.range = [-10, 10]
```

Commands alongside current `to_plotly_json` result with annotations

## Validation Example
Diagram including:
1. Schema snippet (string enumeration maybe?)
2. code: figure.layout.xaxis.bogus = 1
3. code: figure.layout.xaxis.valid = invalid // error message
4. code: figure.layout.xaxis.valid? -> docstring

## Documentation Example
Popup docs for property and constructor

## FigureWidget message diagram
Python code for property assignment ([B]) from Python to JavaScript model and 
views.

## FigureWidget message generation
Maybe a table with label, method, and method data

### construction
Reference [A] and include message data

### assignment
Reference [B] and include message data 

### add traces
Reference [C] and include message data

### reorder traces
Reference [D] and include message data

### delete traces
Reference [E] and include message data

### batch update
Reference [F] and include message data

or pull code out and just talk about it here

### batch animate
Reference [G] and include message data

or pull code out and just talk about it here


## Big zoom example
  - Code (constructor and callback)
  - Zoom
  - Message diagram
  - Callback execution

# Default properties
newPlot() ->
<- traceDefaults()

Examples
--------
1) Animation
2) Brushing
3) Resample 2d histogram
