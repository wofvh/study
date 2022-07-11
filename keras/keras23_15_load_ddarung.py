from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 #pandas : 엑셀땡겨올때 씀
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (715, 9)

print(train_set.columns)
print(train_set.info())                                     # info 정보출력
print(train_set.describe())                                 # describe 평균치, 중간값, 최소값 등등 출력

# 결측치 처리 1. 제거 #############

print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.mean())       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(test_set.mean()) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값
print(train_set.isnull().sum()) 
print(train_set.shape)   # (1328, 10)

############################            


x = train_set.drop(['count'], axis=1)                    # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (1459, 9)

y = train_set['count'] 
print(y)
print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=58525
                                                    )
scaler = MaxAbsScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

#2. 모델구성
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=9))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(70, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(1))

# input1 = Input(shape=(9,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(10)(input1)          # Dense 뒤에 input 부분을 붙여넣는다.
# dense2 = Dense(100, activation='relu')(dense1)
# dense3 = Dense(80, activation='relu')(dense2)
# dense4 = Dense(70, activation='relu')(dense3)
# dense5 = Dense(50, activation='relu')(dense4)
# dense6 = Dense(15, activation='relu')(dense5)
# output1 = Dense(1, activation='relu')(dense6)

# model = Model(inputs = input1, outputs = output1)

import time
start_time = time.time()

#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, 
                              restore_best_weights=True)

# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[earlyStopping])
#model.save("./_save/keras23_15_load_ddarung.h5")
model = load_model("./_save/keras23_15_load_ddarung.h5")
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x, y) 


y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)

print("걸린시간 :",end_time)


# loss :  [76668336.0, 8033.75341796875]
# RMSE :  47.74230472724731     
# r2스코어 :  0.6412159987931108
# 걸린시간 : 7.778677463531494

# save 한 후
# loss :  [76668336.0, 8033.75341796875]
# RMSE :  47.74230472724731     
# r2스코어 :  0.6412159987931108
# 걸린시간 : 0.1805589199066162 