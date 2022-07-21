# from click import argument
# from sklearn.utils import shuffle
from colorsys import yiq_to_rgb
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D ,Flatten, Dense, Dropout
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

train_dataen = ImageDataGenerator(               
    rescale=1./255,                             
                                                
    horizontal_flip=True,                       
    # vertical_flip=True,                         
    width_shift_range=0.1,                      
    height_shift_range=-0.1,                       
    rotation_range=5,                          
    zoom_range=0.1,                            
    # shear_range=0.7,                            
    fill_mode='nearest'                         
)

augument_size = 40000                            # 반복횟수
randidx =np.random.randint(x_train.shape[0],size=augument_size)

print(x_train.shape[0])         # 60000
print(y_train.shape[0])         # 60000
print(x_test.shape[0])          # 10000
print(y_test.shape[0])          # 10000
print(randidx.shape)            # 40000
print(randidx)                  # [39683 20510 12895 ... 24908 55852  1491] 
print(np.min(randidx),np.max(randidx))      # random 함수 적용가능. 
print(type(randidx))            # <class 'numpy.ndarray'> 기본적으로 리스트 형태.       


x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

print(x_augumented.shape)       # (40000, 28, 28)
print(y_augumented.shape)       # (40000,)

x_train = x_train.reshape(60000,28,28,1)


x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)


x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2],
                                    1)
x_augumented = train_dataen.flow(x_augumented,y_augumented,
                                batch_size = augument_size,
                                shuffle=False).next()[0]
print(x_augumented)
print(x_augumented.shape)
print(x_train.shape)            # (100000, 28, 28, 1)

x_train =np.concatenate((x_train,x_augumented))
print(x_train.shape)            # (100000, 28, 28, 1)
y_train =np.concatenate((y_train,y_augumented))
print(y_train.shape)            # (100000,)

print(x_train.shape,y_train.shape)
print(x_train[0].shape)                         # (28, 28)
print(x_train[0].reshape(28*28).shape)          # (784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape)          # (100, 28, 28, 1)
# reshape  # (100, 28, 28, 1) (열, reshape,reshape,reshape)

print(np.zeros(augument_size))
print(np.zeros(augument_size).shape)
print(np.tile(x_train[0].reshape(28*28), augument_size).shape)                      # (31360000,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape)  # (40000, 28, 28, 1)

y_train = pd.get_dummies((y_train))
y_test = pd.get_dummies((y_test))

# x_data = train_dataen.flow(
#     np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),   # x
#     np.zeros(augument_size),                                                 # y
#     batch_size=augument_size,
#     shuffle=True)#.next()   # < 알아보기 
# ##############################next사용 ###################################
# # print(x_data)
# # print(x_data[0])
# # print(x_data[0].shape)               # (100, 28, 28, 1)
# # print(x_data[1].shape)               # (100,)
# ##############################next 미사용 #####################################
# print(x_data)
# print(x_data[0][0])
# print(x_data[0][0][0].shape)               # (100, 28, 28, 1)
# print(x_data[0][0][1].shape)               # (100,)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49) :
#     plt.subplot(7,7,i+1)
#     plt.axis('off')
#     # plt.imshow(x_data[0][i], cmap='gray')        # next사용
#     plt.imshow(x_data[0][0][i], cmap='gray')       # next미사용
    
# plt.show()
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5),   
                 padding='same',
                 input_shape=(28, 28, 1)))                                 
model.add(MaxPooling2D())                                               
model.add(Conv2D(32, (2,2), activation= 'relu'))                                                            
model.add(Flatten())                        
model.add(Dense(16, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(8, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일 구성 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
              verbose=1, restore_best_weights = True)     
        
hist = model.fit(x_train, y_train, epochs=100, batch_size=320,verbose=2,
                 validation_split=0.2, callbacks=[earlystopping])

#4. 평가, 예측\
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

y_predict = model.predict(x_test)
# print('y_predict: ',y_predict[-1])

y_predict = tf.argmax(y_predict,axis=1) 
# print(y_predict)
y_test = tf.argmax(y_test,axis=1)         # argmax 형태가 맞지만, 값이 너무 달라 비교가 안될때 사용.
                                          # [6,7,9,10]   >> 3 반환. (0123 순서로 계산.)
                                          # [3,8,1,2]    >> 1 반환. 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)
