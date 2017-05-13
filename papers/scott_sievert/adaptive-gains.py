import numpy as np
import matplotlib.pyplot as plt
#  plt.style.use('ggplot')

t = np.linspace(0, 1, num=1000)
slow = 1 - np.exp(-2*t)
fast = 1 - np.exp(-6*t)

plt.figure(figsize=(5, 5))
plt.plot(slow, label='Passive')
plt.plot(fast, label='Adaptive')
plt.xlabel('Number of examples')
plt.ylabel('Quality')
plt.legend(loc='best')
#  plt.grid()
plt.savefig('figures/adaptive-gains.png')
plt.show()
