# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###Import some libraries.  Set some constants.

# <codecell>

# this places the plots inline in the notebook
%matplotlib inline  
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt

# <markdowncell>

# ###Define next_radix function.  I could not find capability in scipy/numpy.  

# <codecell>

def next_radix(n):
    """
    Compute the smallest integer larger then n that factors into a 
    product of 2, 3, and 5.  This is used for a mixed radix fft that is 
    faster than an arbitrary length fft.  This is done my trying any i,
    j, and k vaues to find the smallest 2**i * 3**j * 5**k.

    input parameters:
        n - request is for mixed radix greater or equal to this integer
    return values:
        fft_length - recommended radix for fft.
    """
    min_exceeding_n=n*2+1
    for i in range(0,15):
        for j in range(0,15):
            for k in range(0,15):
                test=2**i*3**j*5**k
                if test>=n: # test is large enough, so break out of loops
                    # if test is smaller then all previous candidates
                    if test<min_exceeding_n:
                        min_exceeding_n=test
                    break
    return min_exceeding_n

# <markdowncell>

# ###The models are created by putting spikes at time sample 51 and band limitting by zeroing the high frequencies in the Fourier domain
# ####Set model size.  Initialize ramp for frequency limitting.  
# Frequency domain ramp is:
# <code>
# Frequency     Scale
#  0            1
# .4 nyquist    1
# .6 nyquist    0
# nyquist       0
# </code>
#   
# Display the ramp.

# <codecell>

#define size of test section
nx=32
dt=.004
tmax=.400

# derived parameters
nt=int(tmax/dt)+1
fft_length = next_radix(nt+125) # minimum pad is 125 points or .5s at 4ms
# make the frequency domain ramp that scales 
# f=0          by 1
# f=.4 nyquest by 1
# f=.6 nyquist by 0
# f=nyquist by 0

nf=fft_length/2+1   # number of frequencies after real to complex fft
rampstart=int(round(.4*nf))
rampend  =int(round(.6*nf))
ramplen  =rampend-rampstart+1
nf=fft_length/2+1
ramp=np.zeros(nf)
ramp[0:rampstart]=1
ramp[rampstart:rampstart+ramplen]=np.linspace(1.0,0.0,ramplen)
ramp[rampstart+ramplen:]=0
#plt.figure()
plt.plot(ramp,'r')
#plt.show()
plt.savefig('ramp.png')
plt.show()

# <markdowncell>

# ####Create the signal, noise, and data.  
# Signal is spike at .2 s increasing by 5%/trace.  Bandlimitted in frequency domain.  
# Noise is spike at .2 s constatnt amplitude. banns limitted in frequency domain.  
# Data is sum of signal and noise.
# 
# Display all three sections (signal, noise, data).

# <codecell>

#signal plane: spike at .2s amplitude increases with trace number
s=np.zeros((nt,nx))
s[51,:]=(1.05)**np.linspace(0.0,31.0,32)

#noise plane: spike at .2s amplitude constant with trace
n=np.zeros((nt,nx))
n[51,:]=1.0

#apply bandpass filter in frequency domain
# forward fft
S=np.fft.rfft(s,n=fft_length,axis=0)
# multiply ramp onto signal in frequency domaim
S*=ramp[:,np.newaxis]
# inverse fft. extra [:nt,:] gets rid of fft padding
s_filt=(np.fft.irfft(S,n=fft_length,axis=0))[:nt,:]

# repeat the frequency domain bandpass filter on the noise
N=np.fft.rfft(n,n=fft_length,axis=0)
N*=ramp[:,np.newaxis]
n_filt=(np.fft.irfft(N,n=fft_length,axis=0))[:nt,:]

# data is sum of filtered signal and filtered noise
D=S+N
d=s_filt+n_filt

# plot signal, noise, and data side by side.  sync the zoom & scroll
view1=plt.subplot(1,3,1)
plt.imshow(s_filt,aspect='auto')

view2=plt.subplot(1,3,2,sharex=view1,sharey=view1)
plt.imshow(n_filt,aspect='auto')

view3=plt.subplot(1,3,3,sharex=view1,sharey=view1)
plt.imshow(d,aspect='auto')
plt.savefig('model.png')
plt.show()

# <markdowncell>

# ####Compute the noise model, the noise section with a derivative filter 

# <codecell>

# compute the noise model.  figure 2a in Spitz paper.  Just make it
# the negative of the derivative filter on the noise
m=sg.lfilter(np.array([-1.0,1.0]),             # filter (numerator)
             np.array([1.0]),                   # recursive filter (denominator)
             n_filt,axis=0)                    # the filtered noise section

M=np.fft.rfft(m,n=fft_length,axis=0)           # fourier domain

view1=plt.subplot(1,2,1)
plt.imshow(n_filt,aspect='auto')

view2=plt.subplot(1,2,2,sharex=view1,sharey=view1)
plt.imshow(m,aspect='auto')
plt.savefig('noisemodel.png')
plt.show()

# <markdowncell>

#  ####Compute noise prediction filter, pfa
# pfa is the prediction filter from the multiple model.  Each point in M is predicted by the previous value scaled by pfa0. We want to get to best fit to the equations:
#  <code>
#      M0              pfa0           M1
#      M1                             M2
#      M2         *           =       M3 
#      ...                            ...  
#      Mn-1                           Mn
# </code>
# The prediction error filter is the error made by the prediction filter.  The prediction error filter is 1 followed by the sign reversed prediction error filter.
# 
# To continue in the next code fragment, solve this as a general general matrix problem. Do not compute a simple projection.

# <codecell>

# work at one frequency for now
ifreq=int(round(nf/4.0))

# compute pefa, the prediction error filter from the noise (multiple)model
Aa=np.matrix([M[ifreq,0:-1]]).transpose()
ba=np.matrix(M[ifreq,1:]).transpose()
# multiply both sides by A.transpose.conj
AactAa=Aa.conj().transpose() * Aa
Aactba=Aa.conj().transpose() * ba
pfa=sp.linalg.solve(AactAa,Aactba,sym_pos=True)
pefa=np.zeros(2,dtype=complex)
pefa[0]=1.0
pefa[1:]=-pfa[:,0]
#print "pfa=",pfa
print "pefa=",pefa

# <markdowncell>

# Spitz, paper says pefa should be (1,-1).  This code will produce complex numbers so the filter is pefa= [ 1.+0.j -1.-0.j].

# <markdowncell>

# ####Compute data prediction filter, pfb
# pfb is the prediction filter from data.  Each point in D is predicted by the sum of the two 
# previous value scaled by pfb0 and pfb1. 
# We want to get to best fit to the equations:
# <code>
#      D1   D0              pfb0          D2
#      D2   D1              pfb1          D3
#      D3   D2        *           =       D4 
#      ...                            ...  
#      Dn-1 Dn-2                          Dn
# </code>
# The prediction error filter is the error made by the prediction filter.  The prediction error filter is 1 followed by the sign reversed prediction filter.

# <codecell>

# Now do it with the 2 point pfb. The prediction filter for the data.
# The data has two events (the expoential signal), and the noise (the constant
# multiple).
# the equations are:
#     D1    D0        pf0           D2
#     D2    D1        pf1           D3
#     D3    D2   *           =      D4
#     ...                           ...
#     Dn-1  Dn-2                    Dn
#
Ab=np.matrix([D[ifreq,1:-1],D[ifreq,0:-2]]).transpose()
bb=np.matrix(D[ifreq,2:]).transpose()
# multiply both sides by A.transpose.conj
AbctAb=Ab.conj().transpose() * Ab
Abctbb=Ab.conj().transpose() * bb

pfb=sp.linalg.solve(AbctAb,Abctbb,sym_pos=True)
pefb=np.zeros(3,dtype=complex)
pefb[0]=1.0
pefb[1:]=-pfb[:,0]


print "pefb=",pefb

# <markdowncell>

# Spitz' paper says pefb should be (1, -2.05, 1.05).  This code will produce complex numbers, so the filter is pefb= [ 1.00+0.j -2.05-0.j  1.05-0.j].

# <codecell>

#pefc is pef for the signal.  It is computed by deconvolving pfeb with pefa
# and taking the first two points

pefc=sg.lfilter(np.array([1.0]),pefa,pefb)[:2]

print "pefc=",pefc


# <markdowncell>

# Spitz' paper says pefa should be (1,-1.05).  This code will produce complex numbers, so the filter should be pefc= [ 1.00+0.j -1.05+0.j].

# <markdowncell>

# Fit the data with a linear combination of 1/pefb and 1/pefc.

# <codecell>

impulse=np.zeros(nx,dtype=complex)
impulse[0]=1.0
one_over_pefa=sg.lfilter(np.array([1.0]),pefa,impulse)
plt.plot(np.real(one_over_pefa))
plt.show()

one_over_pefc=sg.lfilter(np.array([1.0]),pefc,impulse)
plt.plot(np.real(one_over_pefc))
plt.show()

F=np.matrix([one_over_pefa,one_over_pefc]).transpose()
# multiply both sides by A.transpose.conj
FctF=F.conj().transpose() * F
Fctd=F.conj().transpose() * np.matrix(D[ifreq,:]).transpose()

coefficients=sp.linalg.solve(FctF,
                    Fctd,sym_pos=True)
print 'D=',D[ifreq,:6]
print coefficients

# <markdowncell>

# The plots above show the two patterns (basis vectors) used to fit the data.  The first few points of the data D are printed and the weights for the two patterns.  It is simple to verify the first point is fit.  Remembering the model verifies the D is fit.  

# <codecell>

def estimate_pef(D, nevents):
    """
    Estimate a prediction filter of length nevents.  Return prediction error filter (pef) 
    length nevents+1
    """
    list_of_columns=[]
    for icol in range(nevents):
        list_of_columns.insert(0,D[icol:icol-nevents])
  
    An=np.matrix(list_of_columns).transpose()
    bn=np.matrix(D[nevents:]).transpose()
    AnctAn=An.conj().transpose() * An
    Anctbn=An.conj().transpose() * bn
    # add small number to diagonal to avoid zero matrix
    # probably need 1% whitenoise for general singular pblms
 
    for irow in range(nevents):
        AnctAn[irow,irow]+=1e-31
        
    pfn=sp.linalg.solve(AnctAn,Anctbn,sym_pos=True)

    pef=np.zeros(nevents+1,dtype=complex)
    pef[0]=1
    pef[1:]=-pfn[:,0]  
    return pef

newpefb=estimate_pef(D[ifreq],2)
print "new pefb=\n",newpefb
        
print "old pefb=\n",pefb
newpefa=estimate_pef(M[ifreq],1)
newpefc=sg.lfilter(np.array([1.0]),newpefa,newpefb)[:2]
print "newpefa=",newpefa
print "newpefc=",newpefc

# <codecell>

allpefa=np.zeros((2,nf))
allpefb=np.zeros((3,nf))
allpefc=np.zeros((2,nf))
for indxfreq in range(nf):
    #print "indxfreq=",indxfreq
    allpefa[:,indxfreq]=estimate_pef(M[indxfreq],1)
    allpefb[:,indxfreq]=estimate_pef(D[indxfreq],2)
    allpefc[:,indxfreq]=sg.lfilter(np.array([1.0]),allpefa[:,indxfreq],allpefb[:,indxfreq])[:2]
   
viewpefa=plt.subplot(1,3,1)
plt.imshow(np.real(allpefa),aspect='auto')
viewpefb=plt.subplot(1,3,2)
plt.imshow(np.real(allpefb),aspect='auto')
viewpefc=plt.subplot(1,3,3)
plt.imshow(np.real(allpefc),aspect='auto')
plt.show()

# <codecell>

impulse=np.zeros(nx,dtype=complex)
impulse[0]=1.0
Sestimate=np.zeros((nf,nx),dtype=complex)
Nestimate=np.zeros((nf,nx),dtype=complex)
for indxfreq in range(nf):

    one_over_pefa=sg.lfilter(np.array([1.0]),allpefa[:,indxfreq],impulse)
    one_over_pefc=sg.lfilter(np.array([1.0]),allpefc[:,indxfreq],impulse)

    F=np.matrix([one_over_pefa,one_over_pefc]).transpose()
    # multiply both sides by A.transpose.conj
    FctF=F.conj().transpose() * F
    Fctd=F.conj().transpose() * np.matrix(D[indxfreq,:]).transpose()

    weights=sp.linalg.solve(F.conj().transpose() * F,
                            F.conj().transpose() * np.matrix(D[indxfreq,:]).transpose(),sym_pos=True)
    Nestimate[indxfreq,:]=weights[0]*one_over_pefa
    Sestimate[indxfreq,:]=weights[1]*one_over_pefc
    

# forward fft
#S=np.fft.rfft(s,n=fft_length,axis=0)
# inverse fft. extra [:nt,:] gets rid of fft padding
sestimate=(np.fft.irfft(Sestimate,n=fft_length,axis=0))[:nt,:]
nestimate=(np.fft.irfft(Nestimate,n=fft_length,axis=0))[:nt,:]
#print sestimate
view_sest=plt.subplot(1,2,1)
plt.imshow(sestimate,aspect='auto')
view_nest=plt.subplot(1,2,2)
plt.imshow(nestimate,aspect='auto')
plt.show()

# <codecell>

for indxfreq in range(nf):
    #print "indxfreq=",indxfreq
    pefal=estimate_pef(M[indxfreq],1)
    pefbl=estimate_pef(D[indxfreq],2)
    pefcl=sg.lfilter(np.array([1.0]),pefal,pefbl)[:2]
    one_over_pefa=sg.lfilter(np.array([1.0]),pefal,impulse)
    one_over_pefc=sg.lfilter(np.array([1.0]),pefcl,impulse)

    F=np.matrix([one_over_pefa,one_over_pefc]).transpose()
    # multiply both sides by A.transpose.conj
    FctF=F.conj().transpose() * F
    Fctd=F.conj().transpose() * np.matrix(D[indxfreq,:]).transpose()

    weights=sp.linalg.solve(FctF, Fctd, sym_pos=True)
    Nestimate[indxfreq,:]=weights[0]*one_over_pefa
    Sestimate[indxfreq,:]=weights[1]*one_over_pefc
    
# inverse fft. extra [:nt,:] gets rid of fft padding
sestimate=(np.fft.irfft(Sestimate,n=fft_length,axis=0))[:nt,:]
nestimate=(np.fft.irfft(Nestimate,n=fft_length,axis=0))[:nt,:]
#print sestimate
view_sest=plt.subplot(1,2,1)
plt.imshow(sestimate,aspect='auto')
view_nest=plt.subplot(1,2,2)
plt.imshow(nestimate,aspect='auto')
plt.savefig('seperatedcomponents.png')
plt.show()

# <codecell>


# <codecell>


