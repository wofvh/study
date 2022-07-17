from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 #pandas : 엑셀땡겨올때 씀
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from tensorflow.python.keras.callbacks import EarlyStopping
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

print(x_train.shape,x_test.shape)  #(1167, 9) (292, 9)

x_train = x_train.reshape(1167, 3,3)
x_test = x_test.reshape(292, 3,3)


print(x_train.shape,x_test.shape)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(LSTM(256, input_shape=(3,3)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()    



#3. 컴파일, 훈련

model.compile(loss= 'mae', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='auto',
                             restore_best_weights=True,verbose=1)

import time
start_time = time.time()

model.fit(x_train, y_train, epochs=500, batch_size=50,
          validation_split=0.2,callbacks=[earlyStopping], verbose=1) #batch default :32
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 


y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print('시간 :' , end_time)
#4. MaxAbsScaler
# loss :  [2539560448.0, 45863.0859375]
# RMSE :  44.13384823645532
# r2스코어 :  0.6934015870340235
# 걸린시간 : 17.524970531463623

# LSTM
# loss :  [35.138092041015625, 0.0034246575087308884]
# RMSE :  51.30793955249752
# r2스코어 :  0.585623176601102
# 시간 : 37.798224210739136
