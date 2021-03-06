from tabnanny import verbose
from sqlalchemy import false, true
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
from sklearn.metrics import r2_score 

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.95, shuffle=True, random_state=77)

print (x)
print (y)
print(x.shape, y.shape)   #(506. 13) (506,)

print(datasets.feature_names)
print(datasets.DESCR)

# # [실습] 아래를 완성하기
# 1. train 0.7
# 2. R2 0.8 이상


# 2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(98))
model.add(Dense(72))
model.add(Dense(85))
model.add(Dense(95))
model.add(Dense(82))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='rmse', optimizer='adam')
model.fit(x_train, y_train, epochs = 150
          , batch_size=35, verbose=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


#R2
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)