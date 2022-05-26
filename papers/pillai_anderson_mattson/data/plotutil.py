import matplotlib.pyplot as plt
import numpy as np

def plot_a(x1, y, leg, log=False,f='test', s='30000', yl='MB/s', xl='nodes', suff='1.png', t=['1'], prescale=1.0, title=None, ylim=None, c=None, ax=None, savefig=True):
    x0 = [i+1 for i in range(len(x1))]
    if ax is None:
        fig = plt.figure(figsize=[4.8,3.6])
        ax = fig.add_axes([0,0,1,1])#,ylabel=yl, xlabel=xl)
    if log: plt.yscale('log')
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if title is None: title=f
    if ylim is not None:
        ax.set_ybound(upper=ylim)
        ax.set_autoscaley_on(False)
    #x0, y, leg, xl = plotdata(f=[f], s=[s], t=t, c=c)
    x = np.asarray(x0)
    xw = 0.8/len(leg)
    x = x - xw*(len(leg)-1)/2
    #plt.axes(yscale='log',xlabel='code size in bytes',ylabel='seconds')
        # ybound=[0.001,50], autoscaley_on=False)
    for i,l in enumerate(leg):
        plt.bar(x+i*xw,np.asarray(y[i])*prescale,xw,label=l)
    ax.set_xticks(x0)
    ax.set_xticklabels(x1)
    ax.legend(ncol=2, loc='upper left')
    ax.set_title(title)
    #plt.legend(ncol=2, bbox_to_anchor=(1.025,1.02),loc='lower right')
    if savefig: plt.savefig(f+suff, bbox_inches='tight')
    #plt.show()

