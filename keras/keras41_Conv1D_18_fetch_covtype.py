import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import pandas as pd 
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Conv1D, Flatten

#1.데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)

# from sklearn.preprocessing import OneHotEncoder
print('y의 라벨값 :', np.unique(y,return_counts=True))
# y의 라벨값 : (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
    #   dtype=int64))


###########(pandas 버전 원핫인코딩)###############
y_class = pd.get_dummies((y))
print(y_class.shape) # (581012, 7)

# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
#1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)

###########(sklearn 버전 원핫인코딩)###############
#from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# train_cat = ohe.fit_transform(train[['cat1']])
# train_cat


# num = num.shape[0]
# print(num)

# y = np.eye(num)[data]
# print(x)
# print(y)



# print(x.shape, y.shape) #(581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(x,y_class, test_size=0.15,shuffle=True ,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape,x_test.shape)  #(493860, 54) (87152, 54)

x_train = x_train.reshape(493860, 18,3)
x_test = x_test.reshape(87152, 18,3)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.

print(y_train.shape,y_test.shape)

#2.모델
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model = Sequential()
# model.add(LSTM(10, input_shape=(3,1), return_sequences =False))     
model.add(Conv1D(128, 2, input_shape=(18,3)))
model.add(Flatten())
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(7, activation='softmax'))


import time
#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
#분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.


# import datetime
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')           # 0707_1723
# print(date)
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint 
# filepath = './_ModelCheckPoint/8fetchcovtpye/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # f > 소수점4자리까지 표현.           

# earlystopping =EarlyStopping(monitor='loss', patience=100, mode='min', 
#               verbose=1, restore_best_weights = True)     
        
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,               # mode acc > max 
#                       save_best_only=True,                                      # patience 필요없음.
#                       filepath ="".join([filepath,'8fetchcovtpye_',date, '_', filename])
#                       ) 
start_time = time.time()

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',
                             restore_best_weights=True,verbose=1)


model.fit(x_train, y_train, epochs=100, batch_size=2000,
          validation_split=0.2,callbacks=[earlyStopping], verbose=1) #batch default :32

end_time = time.time() - start_time


#4.평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
#################### 위와 동일###############
results = model.evaluate(x_test,y_test)
print('loss : ', results[0])
# print('accuracy : ', results[1])
############################################

# print(y_test)
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict,axis=1) 

# print(y_test)
# print(y_test.shape)

y_test = tf.argmax(y_test,axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

print(y_predict)
print(y_test)
print("걸린시간 :",end_time)


# loss :  0.6954039931297302
# acc :  0.7074421700018358
# 걸린시간 : 11.360329389572144

# Conv1D 
# loss :  0.36007747054100037
# acc :  0.8601523774554801
# 걸린시간 : 97.52586221694946