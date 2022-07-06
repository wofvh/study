

from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/gulim.TTc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
print(datasets.DESCR)
print(datasets.feature_names)
print(x)
print(y)
print(x.shape,y.shape) # (150, 4) (150,)
print("y의 라벨값 : ", np.unique(y))  # y의 라벨값 :  [0 1 2]
y = to_categorical(y) # https://wikidocs.net/22647 케라스 원핫인코딩
# print(y)
# print(y.shape) #(150, 3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))



#2. 모델

model = Sequential()
model.add(Dense(30, input_dim=4, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dense(20, activation='sigmoid'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
model.add(Dense(20, activation='relu'))               # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
model.add(Dense(20, activation='linear'))               
model.add(Dense(3, activation='softmax'))             # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
                                                                 # ex 70, 20, 10 -> 0.7, 0.2, 0.1
import time

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)   

start_time = time.time()

model.fit(x_train, y_train, epochs=200, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time() - start_time


#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)
print("걸린시간 : ", end_time)
results= model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)


y_predict = np.argmax(y_predict, axis= 1)

y_predict = to_categorical(y_predict)


acc= accuracy_score(y_test, y_predict) 
print('acc스코어 : ', acc) 

# loss :  0.0530550517141819
# accuracy :  1.0





#1. scaler 하기전 
# loss :  0.07969877123832703
# accuracy :  1.0

#2. minmaxscaler
# loss :  0.09077339619398117
# acc스코어 :  0.9666666666666667
# 걸린시간 :  8.01612901687622

#3. standardscaler 
# loss :  0.06145345792174339
# accuracy :  1.0
# 걸린시간 :  7.872552871704102

#4. MaxAbsScaler


#5. RobustScaler
