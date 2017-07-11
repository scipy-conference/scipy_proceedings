import numpy as np
import matplotlib.pyplot as plt

def textbook(**kwargs):
    plt.tick_params(width=0)
    plt.xticks([])
    plt.yticks([])

t = np.linspace(0, 1, num=1000)
slow = 1 - np.exp(-2*t)
fast = 1 - np.exp(-8*t)

n = np.linspace(0, 7.5, num=1000) + 1
passive = n - 1
adaptive = np.log(n)

#  n = np.arange(80) + 1
#  passive = n**2
#  adaptive = n * np.log(n)

width = 4
ratio = 8 / 5
ratio = 1
plt.figure(figsize=(width, width * ratio))

#  plt.subplot(2, 1, 1)
textbook()
plt.plot(n, passive, label='Passive')
plt.plot(n, adaptive, label='Adaptive')
plt.xlabel('Problem difficulty →')
plt.ylabel('Data collection cost →')
#  plt.title('Collection cost for different ')
plt.legend(loc='best')

#  plt.subplot(2, 1, 2)
#  textbook()
#  plt.plot(slow, label='Passive')
#  plt.plot(fast, label='Adaptive')
#  plt.xlabel('Number of responses →')
#  plt.ylabel('Prediction accuracy →')
#  plt.title("One model's quality")
#  plt.legend(loc='best')

plt.tight_layout()
plt.savefig('figures/adaptive-gains.png')
#  plt.show()
