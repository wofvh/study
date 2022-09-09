import numpy as np
import matplotlib.pyplot as plt
from torch import arange

def relu(x):
    return np.maximum(0,x)

relu2 = lambda x : np.maximum(0,x)

x= arange(-5,5,0.1)
y= relu2(x)


plt.plot(x,y)
plt.grid()
plt.show()


# elu,selu,reaky relu 