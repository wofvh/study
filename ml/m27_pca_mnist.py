from matplotlib.pyplot import axis
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist 
import tensorflow as tf
(x_train,_),(x_test,_) =mnist.load_data()       # _를 사용하면 사용하지않겠다. 

print(x_train.shape,x_test.shape)   # (60000, 28, 28) (10000, 28, 28)
 
x = np.append(x_train,x_test,axis=0)
print(x.shape)                      # (70000, 28, 28)

###############################################################################
#[실습]
# pca를 통해 0.95 이상인 n_vomponents는 몇개? 
# 0.95 0.99 0.999 1 힌트 np. argmax 

# 1 = 78   
###############################################################################

# reshape 784
# cumsum 값을 확인 후에 0.95 이후일 때 몇개인지 1개가 몇개 인지 알아보기 
# 압축했을때 0인 것은 삭제. 

x= x.reshape(70000, 28*28)   

pca = PCA(n_components=784)               # 주성분 / 열축소 13 > 2개로 압축. 
x = pca.fit_transform(x)
# print(x.shape,y.shape )

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)                              # 새로 생긴 피쳐에 값에 중요도를 보여줌. 
print(sum(pca_EVR))                         # 0.999998352533973 모든값을 합친 값.

cumsum = np.cumsum(pca_EVR)             
print(cumsum)
# cumsum 누적합 값이 쌓이면서 1이 된다. 
# [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791   
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835]

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

print(np.argmax(cumsum >=0.95) + 1)         # 154   0.95 이상 값의 열
print(np.argmax(cumsum >=0.99) + 1)         # 331   0.99 이상 값의 열
print(np.argmax(cumsum >=0.999) + 1)        # 486   0.999 이상 값의 열
print(np.argmax(cumsum >=1.0) + 1)          # 713   1.0 이상 값의 열


# print(np.argwhere(cumsum>=0.95)[0])   # [153] 
# print(np.argwhere(cumsum>=1)[0])      # [712]
# # print(tf.argmax(cumsum,0)) # [712]





