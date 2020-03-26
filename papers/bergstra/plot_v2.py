"""Code to plot graphs of benchmarking for scipy paper
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path

def figure(values, names, xticks, cols,
         vertical_lines = [], displacement = 0.0, save=None):
    """Bar plot of values and names    """
    plt.clf()
    barGroups(np.array(values), names, xticks, cols, vertical_lines, displacement)
    plt.ylabel('Data entries per second')

    if save:
        plt.savefig(save, format="svg")


def barGroups(A, names,xticks, colors, vertical_lines, displacement):
    width = 1.0 / (A.shape[0] + 1)
    ind = np.arange(A.shape[0])
    bars = []
    meanA = np.mean(A)
    for p in range(A.shape[0]):
        if p in vertical_lines:
            plt.axvline(x = p, ymin =0, ymax = 1)
        bars += [plt.bar(p+0.1, A[p], color =colors[p])]
        plt.text(p+displacement, A[p]+meanA*.01, '%d'%A[p], size=12, color='k')
    # Remove xticks 
    plt.xticks([x[0] for x in xticks], [x[1] for x in xticks], color='k', size=20)
    #plt.xticks([],[])
    plt.ylim(ymax= np.max(A)*1.1)
    print len(bars)
    print names
    plt.legend(bars, names)


if __name__ == '__main__':

    dirs={}

    def parse_bmark(args, dirname, fnames):
        d={}
        for f in fnames:
            if f.endswith('.bmark'):
                lines = open(os.path.join(dirname,f)).readlines()
                for line in lines:
                    line = line.strip()
                    sp = line.split('\t')
                    d[tuple(sp[0:2])] = float(sp[2])
        dirs[dirname]=d
    
    os.path.walk('numpy',parse_bmark, None)
    os.path.walk('scipy',parse_bmark, None)
    os.path.walk('theano',parse_bmark, None)
    os.path.walk('matlab',parse_bmark, None)
    os.path.walk('eblearn',parse_bmark, None)
    os.path.walk('torch5',parse_bmark, None)


    # Plot 1 : Deep MLP  ( only batch size 60 )
    namesDictionary = [\
             ('theano{gpu/float/60}' ,'Theano(GPU)', 'r') \
             , ('matlab{gpu/float/60}' ,'Matlab with GPUmat(GPU)','k') 
             , ('theano{cpu/double/60}' ,'Theano(CPU)','g') \
             , ('torch5{60}'       ,'Torch 5(CPU) C/C++','b')                \
             , ('numpy{60}'       ,'NumPy(CPU)','y')
             , ('matlab{cpu/double/60}' ,'Matlab(CPU)','m')\
             ]
    labels = [ x[1] for x in namesDictionary]
    values = []
    for x in namesDictionary:
        for d in dirs :
            if dirs[d].has_key( ('mlp_784_500_10',x[0]) ):
                values.append(dirs[d][('mlp_784_500_10', x[0])])
    #ticks = [(15000,0,0.8,1.2,2,'GPU'),(15000,2,3.3,3.7,5,'CPU')]
    #ticks = [ (1.0, '<----GPU---->'), (3.5, '<-----------CPU----------->')]
    ticks = [ (1.0, 'GPU'), (4, 'CPU')]
    colors     = [ x[2] for x in namesDictionary ]

    figure(values, labels,  ticks, colors,vertical_lines = [2], displacement = 0.25, save = 'mlp.svg')

    # Plot 2 : CONV
    namesDictionary = [
        ('theano{gpu/float/1}', 'Theano(GPU)','r')
        , ('theano{cpu/float/1}' , 'Theano(CPU)','g')
        , ('torch5'       , 'Torch 5(CPU) C/C++', 'b')
        , ('scipy{cpu/double/1}'       , 'SciPy(CPU)*','y')
        , ('eblearn'        , 'EBLearn(CPU) C/C++','c')
        ]

    labels = [ x[1] for x in namesDictionary]
    values = []
    for x in namesDictionary:
        for d in dirs:
            if dirs[d].has_key( ('ConvLarge',x[0]) ):
                values.append(dirs[d][('ConvLarge',x[0])])
    colors      = [ x[2] for x in namesDictionary ]
    #ticks = [ (0.5, '<----GPU---->'), (2.0, '<-----------CPU----------->')]
    ticks = [ (0.5, 'GPU'), (3, 'CPU')]
    figure(values, labels, ticks, colors, vertical_lines = [1], displacement = 0.4, save='conv.svg')
