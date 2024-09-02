import numpy as np
from rich import print
import matplotlib.pyplot as plt

data = [136, 1300, 593]
labels = ['ITK-Wasm WASI', 'Native ITK Python', 'ITK-Wasm CuCIM']

x = np.arange(len(data))

plt.figure(figsize=(10, 6))
plt.bar(x, data, capsize=5, color='pink', alpha=0.7)
# plt.xlabel('Benchmark')
plt.ylabel('Venv size (MB) - lower is better')
plt.title('Downsample Package Virtual Environment Size')
plt.xticks(x, labels)

# Save the plot as PNG and SVG
plt.savefig('figures/venv_fig.png', format='png')
plt.savefig('figures/venv_fig.svg', format='svg')

# Display the plot
plt.show()