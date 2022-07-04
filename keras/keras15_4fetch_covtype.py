import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype

#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print (x.shape, y.shape)    # (581012 ,54)
print ( np.unique(y,return_counts=True))      
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], 
#  dtype=int64))
    
    
    
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[1])
plt.show()


# 과제 

# 3가지 원핫인코딩 방식을 비교할것 
# 1. pandas의 get_dummies

# 2. tensorflow의 to_categorical

# 3. sklearn의 OneHotEncoder

# 미세한 차이를 정리하시오. 