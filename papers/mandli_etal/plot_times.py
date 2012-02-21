#!/usr/bin/env python
"""
Plots for petclaw-walla paper.  Data is from run data and this script simply
plots the results
"""

import numpy as np
import matplotlib.pyplot as plt



scaling_data = np.array([[1    , 76.7, 98.9,  154], 
                         [4    ,   69,101.1,  152],
                         [16   , 71.7,103.2,  164],
                         [64   , 73.7,103.0,  217],
                         [256  ,   74,103.4,  407],
                         [1024 ,   75,103.9,  480],
                         [4096 , 76.6,104.9,  898],
                         [16384, 79.6,112.9, 3707]])

parallel_efficincy = scaling_data.copy()

parallel_efficincy[:,1] = parallel_efficincy[0,1]/parallel_efficincy[:,1]
parallel_efficincy[:,2] = parallel_efficincy[0,2]/parallel_efficincy[:,2]
parallel_efficincy[:,3] = parallel_efficincy[0,3]/parallel_efficincy[:,3]

walla_data = np.array([[1   , 4.185741e+00, 3.569709e+00],
                       [2   , 4.293594e+00, 5.708786e+00],
                       [4   , 5.307578e+00, 5.384538e+00],
                       [8   , 5.567596e+00, 5.468542e+00],
                       [16  , 8.708066e+00, 8.095164e+00],
                       [32  , 1.275780e+01, 1.578909e+01],
                       [64  , 2.742231e+01, 2.825254e+01],
                       [128 , 2.342731e+01, 2.436656e+01],
                       [256 , 2.915356e+01, 3.030362e+01],
                       [512 , 2.927832e+01, 3.022328e+01],
                       [1024, 4.395370e+01, 4.208520e+01],
                       [2048, 4.629234e+01, 4.177804e+01],
                       [4096, 7.004248e+01, 7.221883e+01],
                       [8192, 1.402816e+02, 1.274446e+02]])
                 

font16 = {  'fontsize':16 }
font18 = {  'fontsize':18 }

fig = plt.figure(1)
ax = fig.add_subplot(111)
acoustics_line = ax.semilogx(scaling_data[:,0],scaling_data[:,1],'rx-')
euler_line = ax.semilogx(scaling_data[:,0],scaling_data[:,2],'bo-')
ax.set_title("Weak Scaling - Component Codes",**font18)
ax.set_xticks(scaling_data[:,0])
ax.set_xticklabels([int(value) for value in scaling_data[:,0]])
ax.set_xlabel("Number of Processes",**font16)
ax.set_ylabel("Time (s)",**font16)
ax.set_xlim([scaling_data[0,0],scaling_data[-1,0]])
plt.legend((acoustics_line,euler_line),('Acoustics','Euler'),loc=2)

plt.savefig("code_scaling_results.pdf")


fig = plt.figure(4)
ax = fig.add_subplot(111)
acoustics_line = ax.semilogx(parallel_efficincy[:,0],parallel_efficincy[:,1],'rx-')
euler_line = ax.semilogx(parallel_efficincy[:,0],parallel_efficincy[:,2],'bo-')
ax.set_title("Parallel Efficiency - Component Codes",**font18)
ax.set_xticks(parallel_efficincy[:,0])
ax.set_xticklabels([int(value) for value in parallel_efficincy[:,0]])
ax.set_xlabel("Number of Processes",**font16)
ax.set_ylabel("Parallel Efficiency",**font16)
ax.set_xlim([parallel_efficincy[0,0],parallel_efficincy[-1,0]])
ax.set_ylim([0,1.4])
plt.legend((acoustics_line,euler_line),('Acoustics','Euler'),loc=2)

plt.savefig("parallel_Efficiency_results.pdf")



fig = plt.figure(2)
ax = fig.add_subplot(111)
total_line = ax.semilogx(scaling_data[:,0],scaling_data[:,3],'ko-')

ax.set_title("Scaling Comparisons - Overall Times" ,**font18)
ax.set_xticks(scaling_data[:,0])
ax.set_xticklabels([int(value) for value in scaling_data[:,0]])
ax.set_xlabel("Number of Processes" ,**font16)
ax.set_ylabel("Time (s)",**font16)
ax.set_xlim([scaling_data[0,0],scaling_data[-1,0]])
# ax.set_ylim([0,50])

plt.savefig("total_scaling_results.pdf")
                 
fig = plt.figure(3)
ax = fig.add_subplot(111)
import_line = ax.semilogx(walla_data[:,0],walla_data[:,1],'rx-')
mpiimport_line = ax.semilogx(walla_data[:,0],walla_data[:,2],'bo-')

ax.set_title("Import Timing Comparisons",**font18)
ax.set_xticks(walla_data[:,0])
ax.set_xticklabels([int(value) for value in walla_data[:,0]])
ax.set_xlabel("Number of Processes",**font16)
ax.set_ylabel("Time (s)",**font16)
ax.set_xlim([walla_data[0,0],walla_data[-1,0]])
# ax.set_ylim([0,50])
plt.legend((import_line,mpiimport_line),("Original","Walla"),loc=2)

plt.savefig("walla_comparison.pdf")

# plt.show()



