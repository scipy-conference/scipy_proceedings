import numpy as np
import pandas as pd
from plotutil import plot_a

X = pd.read_csv('stencil.csv')
systems = X['system'].unique()
print (systems)
nodes = X['nodes'].unique()
nodes.sort()
print (nodes)

# overrride -- don;t plot all of the data, change order
systems = ['numpy', 'dask', 'heat', 'ramba', 'mpi']
legends = ['numpy', 'dask array', 'heat', 'ramba', 'C/mpi']
nodes = [1,2,4,8,16,32]

Xb = X[X['nodes']==1][['system','size']]
print (Xb)

yvals=[]
for s in systems:
    Xs = X[X['system']==s]
    bs = Xb[Xb['system']==s]['size'].iloc[0]
    Xs = Xs[Xs['size']==bs]
    y = []
    for i in nodes:
        data = Xs[Xs['nodes']==i]
        if data.shape[0]==0:
            val = 0
        else:
            val = np.mean(data.iloc[0,4:])
        y.append(val)
    yvals.append(y)
#print (yvals)
plot_a(nodes, yvals, legends, log=True, yl="MFlops/s", title="Stencil Strong Scaling", f="stencil-ss-log")
plot_a(nodes, yvals, legends, log=False, prescale=1e-6, yl="TFlops/s", title="Stencil Strong Scaling", f="stencil-ss")

yvals=[]
for s in systems:
    Xs = X[X['system']==s]
    bs = Xb[Xb['system']==s]['size'].iloc[0]
    y = []
    for i in nodes:
        data = Xs[Xs['nodes']==i]
        if i==1:
            data = data[data['size']==bs]
        else:
            data = data[data['size']!=bs]
        if data.shape[0]==0:
            val = 0
        else:
            val = np.mean(data.iloc[0,4:])
        y.append(val)
    yvals.append(y)
#print (yvals)
plot_a(nodes, yvals, legends, log=True, yl="MFlops/s", title="Stencil Weak Scaling", f="stencil-ws-log")
plot_a(nodes, yvals, legends, log=False, prescale=1e-6, yl="TFlops/s", title="Stencil Weak Scaling", f="stencil-ws")
