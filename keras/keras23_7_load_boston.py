from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target #데이터를 리스트 형태로 불러올 때 함

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=66)




# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(datasets.feature_names)
# print(datasets.DESCR)

#2. 모델구성
# input1 = Input(shape=(13,))          # 컬럼3개를 받아드린다.
# dense1 = Dense(64)(input1)          
# dense2 = Dense(32, activation='relu')(dense1)
# dense3 = Dense(16, activation='relu')(dense2)
# dense4 = Dense(8, activation='relu')(dense3)
# dense5 = Dense(4, activation='relu')(dense4)
# output1 = Dense(1)(dense5)

# model = Model(inputs = input1, outputs = output1)



#3. 컴파일,훈련
# model.compile(loss='mae', optimizer='adam')

# earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='auto', 
#                               verbose=1, restore_best_weights=True)


# model.fit(x_train, y_train, epochs=500, batch_size=32, 
#                 validation_split=0.2,
#                 callbacks=[earlyStopping],
#                 verbose=1)
#model.save("./_save/keras23_9_load_boston.h5")
model = load_model("./_save/keras23_9_load_boston.h5")


# #4. 평가,예측
print("=========================1.기본출력========================")
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)

# loss : 2.1817049980163574
# r2 : 0.8656148032741648

# loss : 2.273927927017212
# r2 : 0.8489662839142227

# loss : 2.273927927017212
# r2 : 0.8489662839142227

