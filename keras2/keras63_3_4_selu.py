
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def selu(z, alpha=1.6732632423543772848170429916717, lamb=1.0507009873554804934193349852946):
    return np.where(z < 0, lamb * alpha * (np.exp(z) - 1), z)

z = np.linspace(-5, 5, 200)
plt.plot(z, selu(z), "b-", linewidth=2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.plot([0, 0], [-2.2, 3.2], 'k-')
plt.grid(True)
plt.title(r"SELU activation function ($\alpha \approx 1.6732$ and $\lambda \approx 1.0507$)", fontsize=14)
plt.axis([-5, 5, -2.2, 3.2])

plt.savefig("selu_plot")
plt.show()