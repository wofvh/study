from sklearn.preprocessing import MinMaxScaler, StandardScaler
#[과제] 속도 비교 
# gpu와 cpuimport numpy as np 

from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import pandas as pd
import numpy as np

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('돈다') 
    aaa = 'gpu'
else : 
    print('안돈다') 
    bbb = 'cpu'



#1.데이터 
datasets = fetch_covtype()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(581012, 54) (581012, )
print(np.unique(y,return_counts=True)) #(array[1 2 3 4 5 6 7],array[211840, 283301,  35754,   2747,   9493,  17367,  20510] )

#텐서플로우
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) 

# 판다스 겟더미
# y= pd.get_dummies(y) #argmax 다르게 하니 돌아감. 이유는 모름

#사이킷런
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=58525
                                                    )

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
# scaler.transform(x_test)
x_test =scaler.transform(x_test)
x_train = scaler.transform(x_train)
print(np.min(x_train))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_train))      # 1
print(np.min(x_test))      # 0   알아서 컬럼별로 나눠준다. 
print(np.max(x_test))

#2.모델
model = Sequential()
model.add(Dense(10, input_dim=54))
model.add(Dense(300,activation ='relu'))
model.add(Dense(400,activation ='relu'))
model.add(Dense(800,activation ='relu'))
model.add(Dense(700,activation ='relu'))
model.add(Dense(7,activation ='softmax')) #소프트맥스는 모든 연산값의 합이 1.0,그중 가장 큰값(퍼센트)을 선택,so 마지막 노드3개* y의 라벨의 갯수
#softmax는 아웃풋만 가능 히든에서 x

import time
#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') #다중분류는 무조건 loss에 categorical_crossentropy
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58) #분류모델에서 셔플 중요! ,false로 하면 순차적으로 나와서 2가 아예 안나옴.

start_time = time.time()

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)


model.fit(x_train, y_train, epochs=10, batch_size=1000,validation_split=0.2,callbacks=earlyStopping, verbose=1) #batch default :32

end_time = time.time() - start_time

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


#1. scaler 하기전 
# loss :  0.2693009674549103
# acc :  0.8896984578667156
# 걸린시간 : 52.980239152908325

#2. minmaxscaler
# loss :  0.3631349205970764
# acc :  0.8439737470167065
# 걸린시간 : 18.23297667503357

#3. standardscaler 
# loss :  0.5262272357940674
# acc :  0.7712158068661649
# 걸린시간 : 18.054161310195923

#4. MaxAbsScaler


#5. RobustScaler
