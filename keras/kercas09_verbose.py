
from tabnanny import verbose
from sklearn. datasets import load_boston        
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

     
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size =0.7,                                
    shuffle=True, random_state =5987)
 
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(1))

import time

#3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')

start_time = time.time()

model.fit(x_train, y_train, epochs =5, batch_size = 2, verbose=1)     #verbose = 0 / 훈련과정이 안보인다   크로그래스바없어짐 =2 

and_time = time.time() - start_time

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)

print('r2 스코어 :', r2)
print("걸린시간 : ", and_time )

'''''
votbose 0                               / 출력 없다. 
vorbose 1       1.0337412357330322      / 잔소리많다. 
vorbose 2       1.22182297706604        / 프로그래스바 없다.
vorbose 3,4,5   1.1551883220672607      / epoch만 나온다. 
'''