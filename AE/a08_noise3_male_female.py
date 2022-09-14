#keras 남자 여자에 noise를 넣어서 기미 주근깨 여드름 제거 !
# predict 첫번째 :
# 두번째 : 본인 사진넣어서 빼기
# 랜덤하게 5개 정도 원본/수정본 빼기 

from matplotlib.pyplot import hist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.model_selection import train_test_split



men2 = np.load('d:/study_data/_save/_npy/keras51_6_men1.npy')
x_train = np.load('d:/study_data/_save/_npy/keras47_4_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_4_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_4_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_4_test_y.npy')

x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)

x_train_noised = np.clip(x_train_noised,a_min=0,a_max=1)
x_test_noised = np.clip(x_test_noised,a_min=0,a_max=1)

print(x_train_noised.shape)  # (2647, 100, 100, 3)
print(x_test_noised.shape)   # (662, 100, 100, 3)

# x_train_noised = x_train_noised.reshape(2647,100*100*3)
# x_test_noised = x_test_noised.reshape(662,100*100*3)

print(x_train.shape)  # (2647, 30000)
print(x_test.shape)   # (662,30000)

#2 모델구성 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,(2,2), input_shape = (100,100,3),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(100,activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(30000,activation='relu'))
    model.summary()
    return model

model= autoencoder(hidden_layer_size=154) # pca의 95% 성능
# model= autoencoder(hidden_layer_size=331) # pca의 99% 성능
# x_train = x_train.reshape(2647,30000)
# x_test = x_test.reshape(662,30000)
# x_train_noised = x_train_noised.reshape(2647,30000)
# x_test_noised = x_test_noised.reshape(662,30000)


print(x_train.shape,x_train_noised.shape)   # (2647, 100, 100, 3) (2647, 100, 100, 3)
print(x_test.shape,x_test_noised.shape)     # (662, 100, 100, 3) (662, 100, 100, 3)


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit(x_train_noised,x_train,epochs=10,validation_split=0.3,verbose=2)
output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig,((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15))= \
    plt.subplots(3,5,figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지 맨위에 그린다.
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28))
    if i ==0 :
        ax.set_ylabel('input',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 넣은 이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28))
    if i ==0 :
        ax.set_ylabel('noise',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])    
    
# 오토인콛더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28))
    if i ==0 :
        ax.set_ylabel('output',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

    