#1. 데이터
from matplotlib.colors import rgb2hex
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
x1_datasets = np.array([range(100), range(301, 401)])                    # 삼성전자 종가, 하이닉스 종가
x1 = np.transpose(x1_datasets)


print(x1.shape)       # (100, 2) (100, 3)

y1= np.array(range(2001,2101)) # 금리 
y2= np.array(range(201, 301))

#데이터가 두개 이상일 때 함수형을 사용해서 계산한다 Sequential은 2개 데이터를 계산할 수 없다. 

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,y1,y2,
                                                    train_size=0.7, 
                                                    random_state=100
                                                    )

print(x1_train.shape,x1_test.shape)       # (70, 2) (30, 2)
print(y1_train.shape,y1_test.shape)       # (70,) (30,)
print(y2_train.shape,y2_test.shape)       # (70,) (30,)
''
# # scaler = RobustScaler()
# scaler = StandardScaler()
# scaler.fit([x1_train,x2_train])
# # scaler.transform(x_test)
# x_test =scaler.transform([x1_test, x2_test])
# x_train = scaler.transform([x1_train, x2_train])
# print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
# print(np.max(x_train))      # 1
# print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
# print(np.max(x_test))

#2. 모델구성

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input ,Dense
# from typing_extensions import concatenate
#3-1 
input1 = Input(shape=(2,))
dense1 = Dense(128, activation= 'relu', name ='ys1')(input1)
dense2 = Dense(64, activation= 'relu', name ='ys2')(dense1)
dense3 = Dense(32, activation= 'relu', name ='ys3')(dense2)
dense4 = Dense(16, activation= 'relu', name ='ys4')(dense3)
output1 = Dense(10, activation= 'relu', name ='out_ys1')(dense4)

# #3-2 
# input2 = Input(shape=(3,))
# dense11 = Dense(1280, activation= 'relu', name ='ys11')(input2)
# dense12 = Dense(640, activation= 'relu', name ='ys12')(dense11)
# dense13 = Dense(320, activation= 'relu', name ='ys13')(dense12)
# dense14 = Dense(160, activation= 'relu', name ='ys14')(dense13)
# output2 = Dense(80, name ='out_ys12')(dense14)

# #3-3
# input3 = Input(shape=(2,))
# dense21 = Dense(1280, activation= 'relu', name ='ys21')(input2)
# dense22 = Dense(640, activation= 'relu', name ='ys22')(dense21)
# dense23 = Dense(320, activation= 'relu', name ='ys23')(dense22)
# dense24 = Dense(160, activation= 'relu', name ='ys24')(dense23)
# output3 = Dense(80, name ='out_ys22')(dense24)

# concatenate
from tensorflow.python.keras.layers import concatenate, Concatenate
mergel = concatenate([output1],name ='mg1')              
merge2 = Dense(1280, activation= 'relu',name ='mg2')(mergel)
merge3 = Dense(640, activation= 'relu', name ='mg3')(merge2)
merge4 = Dense(320, activation= 'relu', name ='mg4')(merge3)
merge5 = Dense(160, activation= 'relu', name ='mg5')(merge4)
merge6 = Dense(80, activation= 'relu', name ='mg6')(merge5)
last_output = Dense(1, name ='last')(merge6)

#3-4 output모델
# output41 = Dense(10)(last_output)
# output42 = Dense(40)(output41)
# last_output2 = Dense(1)(output42)

# output51 = Dense(10)(last_output)
# output52 = Dense(40)(output51)
# output53 = Dense(40)(output52)
# last_output3 = Dense(1)(output53)

mergel1 = concatenate([output1],name ='mg11')        # 단순하게 엮는것.      
merge21 = Dense(1280, activation= 'relu',name ='mg12')(mergel1)
merge31 = Dense(640, activation= 'relu', name ='mg13')(merge21)
merge41 = Dense(320, activation= 'relu', name ='mg14')(merge31)
merge51 = Dense(160, activation= 'relu', name ='mg15')(merge41)
merge61 = Dense(80, activation= 'relu', name ='mg16')(merge51)
last2_output = Dense(1, name ='last2')(merge61)

model =Model(inputs =[input1], outputs= [last_output, last2_output])

model.summary()

# aaa = []                                               
# aaa.append([last_output, last2_output])                     


#3. 컴파일,훈련

model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor='val_loss', patience=80, mode='auto', verbose=1, 
                              restore_best_weights=True) 
        
hist = model.fit([x1_test], [y1_train, y2_train], epochs=500, batch_size=16,verbose=1,
                 validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
loss = model.evaluate([x1_test],[y1_test,y2_test])
# loss2 = model.evaluate([x1_test, x2_test, x3_test], y2_test)

print('loss :', loss)
# print('loss2 :', loss2)


y1_predict, y2_predict = model.predict([x1_test])
print(y1_predict)
print(y2_predict)
from sklearn.metrics import r2_score

r2_1 = r2_score(y1_test,y1_predict)
r2_2 = r2_score(y2_test,y2_predict)

# y1_predict = np.array(y1_predict)
# y2_predict = np.array(y2_predict)
# print(y1_predict.shape)
# print(y2_predict.shape)
# y2_test = np.array(y2_test)
# y1_test = np.array(y1_test)
# print(y2_test.shape)
# y1_test = y1_test.reshape(15,1)
# y2_test = y2_test.reshape(15,1)

# print(y1_test.shape,y2_test.shape)

print('r2_1 스코어 :', r2_1)
print('r2_2 스코어 :', r2_2)

# loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
# y_predict = model.predict([x1_test, x2_test, x3_test])
# # print(y_predict.shape)

# print(y_predict.shape)

# # print([y1_test, y2_test].shape)
# y_test = np.array([y1_test, y2_test])
# print(y_test.shape)
# y_predict = y_predict.reshape(2,15)
# r2 = r2_score(y_test,y_predict)

# print('r2: ',r2)

# r2 스코어 : 0.014686027842552019
# r2 스코어 : -0.0269459417847131

# r2_1 스코어 : 0.911922708938242
# r2_2 스코어 : 0.17277146083472872
