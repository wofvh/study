from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target #데이터를 리스트 형태로 불러올 때 함

x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)

from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(datasets.feature_names)
# print(datasets.DESCR)
print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)

x_train = x_train.reshape(404, 13,1)
x_test = x_test.reshape(102, 13,1)

print(x_train.shape, x_test.shape)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units= 10, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
# 10 = units, 3 = timesteps , 1 = feature 
# units * (feature +bias +units)                    # units를 한번더 해준다. 
# model.add(SimpleRNN(32))                          # RNN은 2차원으로 인식해서 바로 Dense적용가능.
# model.add(SimpleRNN(units=10, input_length =3, input_dim=1))       
# model.add(SimpleRNN(units=10, input_dim=1, input_length =3))    # 가독성 떨어짐                                                 # RNN은 2차원으로 인식해서 바로 Dense적용가능.  
model.add(LSTM(350, input_shape=(3,1)))      # [batch, timesteps(몇개씩 자르는지), feature=1(input_dim)]
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='swish'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(1))
                                         # erorr = ndim=3 3차원으로 바꿔라. 
model.summary()



#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=50, mode='min', 
                              verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath='./_ModelCheckPoint/keras24_ModelCheckpoint3.hdf5'
                    )
model.compile(loss='mae', optimizer='adam')

start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=50, 
                validation_split=0.2,
                callbacks=[earlyStopping,mcp],
                verbose=2  )

model.save('./_save/keras24_3_save_model.h5')

# #4. 평가,예측
print("=========================1.기본출력========================")
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("r2 :",r2)



# loss : 9.402571678161621
# r2 : -0.5285233444774788

# dropout
# loss : 9.012863159179688
# r2 : -0.40762565852655896

###############################
# r2 : -0.006260202196402664

