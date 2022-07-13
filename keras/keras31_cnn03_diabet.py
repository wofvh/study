from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import time
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

#1. 데이터
datasets = load_diabetes ()
x = datasets.data
y = datasets.target #데이터를 리스트 형태로 불러올 때 함

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.7,shuffle=True,random_state=100)

print(x_train.shape,x_test.shape)    #(309, 10) (133, 10)


x_train = x_train.reshape(309, 10,1,1)      
x_test = x_test.reshape((133, 10,1,1) )          

print(x_train.shape)
print(np.unique(y_train, return_counts =True))



# scaler = StandardScaler()
# scaler.fit(x_train) 
# # scaler.transform(x_test)
# x_test = scaler.transform(x_test)
# x_train = scaler.transform(x_train)
# # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
# #       dtype=int64))

# # x_train = x_train.reshape(50000, 32, 32, 3)
# # x_test = x_test.reshape(10000, 32, 32, 3)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # y_train = pd.get_dummies(y_train)
# # y_test = pd.get_dummies(y_test)
# print(x_train.shape)
# print(x_test.shape, x_train.shape)



#2. 모델구성
model = Sequential()
model.add(Conv2D(64, kernel_size=(1, 1),   # 출력(4,4,10)                                       # 자르는 사이즈 (행,렬 규격.) 10= 다음레이어에 주는 데이터
                 padding='same',
                 input_shape=(10,1,1), activation= 'relu'))    #(batch_size, row, column, channels)       # N(장수) 이미지 5,5 짜리 1 흑백 3 칼라                                                                        
model.add(Conv2D(64, (5,5), 
                 padding = 'same',         # 디폴트값(안준것과 같다.) 
                 activation= 'relu'))    # 출력(3,3,7)                                                     
model.add(Flatten()) # (N, 63)
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation= 'linear'))
model.summary()


#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='auto', 
                              verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath='./_ModelCheckPoint/keras24_ModelCheckpoint3.hdf5'
#                     )
model.compile(loss='mae', optimizer='adam')

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1000, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=2  )


# #4. 평가,예측
print("=========================1.기본출력========================")
loss = model.evaluate(x_test, y_test)
print('loss :', loss)



y_predict = model.predict(x_test)
print(x_test.shape,y_predict.shape)


from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)


# r2 : 0.8364753069173615


# cnn
#############################
# loss : 50.241390228271484
# (133, 10, 1, 1) (133, 1)
# r2 : 0.1411814220593065