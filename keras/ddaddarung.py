#데이터 불러오기

path = './_data/ddarung/'

import pandas as pd

train_set = pd.read_csv(path + "train.csv", index_col=0)

#xy로 나누기

# print(train_set.shape)

#결측치 제거

print(train_set.info())
print(train_set.isnull().sum())
train_set = train_set.dropna()
print(train_set.isnull().sum())


x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)

#.모델

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(50, input_dim=9))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일 훈련

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=10, batch_size=10)

#4 평가 예측

loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2= r2_score(y_test, y_predict)
print('r2', r2)

test_set = pd.read_csv(path + 'test.csv', index_col=0)

y_summit = model.predict(test_set)

submission = pd.read_csv(path + 'submission.csv')

submission['count'] = y_summit 

submission.to_csv(path + 'submission.csv', index=False)

