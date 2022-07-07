#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.datasets import load_diabetes
import time
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,train_size=0.7,random_state=66
    )

# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)

print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)




#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(85))
model.add(Dense(100))
model.add(Dense(80,activation='relu'))
model.add(Dense(15))
model.add(Dense(1))

import time
start_time = time.time()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=1, 
                              restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,verbose=1,
                 validation_split=0.2, callbacks=[earlyStopping])

end_time = time.time() - start_time

#4. 평가, 예측\
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print("걸린시간 : ", end_time)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

#1. scaler 하기전 


#2. minmaxscaler
# 걸린시간 :  1657089283.3999755
# loss :  [3070.773681640625, 46.019554138183594]
# r2스코어 :  0.5071282135515177

#3. standardscaler 

# 걸린시간 :  1657089352.2844896
# loss :  [3290.931884765625, 46.64012908935547]
# r2스코어 :  0.47179194447365047

#4. MaxAbsScaler
# 걸린시간 :  14.14804720878601
# loss :  [3318.7451171875, 47.68513488769531]      
# r2스코어 :  0.4673278491206041

#5. RobustScaler
# 걸린시간 :  29.881981372833252
# loss :  [3676.0947265625, 50.16604232788086]      
# r2스코어 :  0.40997168231274317
