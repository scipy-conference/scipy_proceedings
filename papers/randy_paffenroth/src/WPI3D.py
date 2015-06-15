#! /usr/bin/env python
"""Some non-linear data sets for testing and visualization.
"""

from playground.src.lib.nonlinearData import WPISwissRoll
import mayavi.mlab as ml
import numpy as np
size = 5000
np.random.seed(1234)
u = np.random.random(size=[size])
v = np.random.random(size=[size])

uN, vN, x, y, z = WPISwissRoll(u, v)

ml.clf()  
ml.points3d(x, y, z, uN, mode='sphere',
            scale_factor=0.1, scale_mode='none')
ml.savefig('../randy_paffenroth/WPI3D.png')

