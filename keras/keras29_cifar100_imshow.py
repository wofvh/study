import numpy as np
from tensorflow.keras.datasets import cifar100
from keras.datasets import mnist, cifar100 , fashion_mnist


(x_train, y_train), (x_test, y_test) =cifar100.load_data()

print(x_train.shape, y_train.shape)    # (60000, 28, 28) (60000,)

print(x_test.shape, y_test.shape)      # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])


import matplotlib.pyplot as plt
plt.imshow(x_train[5],'gray')
plt.show()