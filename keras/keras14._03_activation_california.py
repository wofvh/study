#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
<<<<<<< HEAD

=======
>>>>>>> 59d0b052ece67fe762a9dc5d3f1af5c8fecb35d1
from sklearn.datasets import fetch_california_housing
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
<<<<<<< HEAD
                                                    random_state=58525
                                                    )

# print(x)
# print(y)
# print(x.shape, y.shape) # (506, 13) (506,)
# print(datasets.feature_names) #싸이킷런에만 있는 명령어
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(15))
=======
                                                    random_state=66
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=8,activation='sigmoid'))
model.add(Dense(30,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10))
>>>>>>> 59d0b052ece67fe762a9dc5d3f1af5c8fecb35d1
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
<<<<<<< HEAD
hist = model.fit(x_train, y_train, epochs=5000, batch_size=80,
=======
hist = model.fit(x_train, y_train, epochs=3000, batch_size=100,
>>>>>>> 59d0b052ece67fe762a9dc5d3f1af5c8fecb35d1
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)

<<<<<<< HEAD

##################val전후#################  
# loss :  [0.5288467407226562, 0.5303353667259216]
# r2스코어 :  0.6033424715970717 
=======
# loss :  0.6447481513023376
# r2스코어 :  0.5096276859675669
##################val전후#################  72
# loss :  0.5981025099754333
# r2스코어 :  0.545104755121719   
>>>>>>> 59d0b052ece67fe762a9dc5d3f1af5c8fecb35d1
##################EarlyStopping전후#################
# loss :  0.5845963358879089
# r2스코어 :  0.5505772840253547
##################activation전후#################
<<<<<<< HEAD
# loss :  [0.49348387122154236, 0.5228223204612732]
# r2스코어 :  0.629866157851986
=======
# loss :  0.45930835604667664
# r2스코어 :  0.6705258342277447
>>>>>>> 59d0b052ece67fe762a9dc5d3f1af5c8fecb35d1
