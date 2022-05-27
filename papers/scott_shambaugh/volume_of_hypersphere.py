import math
import numpy as np
import matplotlib.pyplot as plt


r = 0.5
k = list(range(1, 11))
v1 = [1/np.math.factorial(x) for x in k]
v2 = [np.pi**(x/2)/(2**x * math.gamma(x/2+1)) for x in k]
print(k)
print(v1)
print(v2)

fig, ax = plt.subplots(1,1)
plt.axhline(1, c='k')
ax.semilogy(k, v1, '-*')
ax.plot(k, v2, '--*')

plt.xticks(k)
plt.xlabel('Number of Input Dimensions $k$')
plt.ylabel('Enclosed Volume $V_{(n \\to \\infty)}$')
plt.legend(['Random Sampling', 'OAT $l_1$ norm (cross-polytope)', 'OAT $l_2$ norm (hypersphere)'])

fig.set_size_inches(6.0, 4)
plt.savefig('hypersphere_volume.png', dpi=100)
