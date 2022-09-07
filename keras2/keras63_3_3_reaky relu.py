import numpy as np
import matplotlib.pyplot as plt
from torch import arange

def leakyrelu(x):
    return np.maximum(0.1*x,x)

leakyrelu2 = lambda x : np.maximum(0.1*x,x)

x= arange(-5,5,0.1)
y= leakyrelu2(x)


plt.plot(x,y)
plt.grid()
plt.show()


# elu,selu,reaky relu 