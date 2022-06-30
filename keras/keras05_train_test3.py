import numpy as np
from sqlalchemy import false, true
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라 

from sklearn.model_selection import train_test_split     
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size =0.7,                                # 0.7 =70%
    shuffle=True, 
    random_state =66)

# from sklearn.model_selection import train_test_split    # train_test_split()는 train set과 test set 으로 분할하는 기능.
# x_tratin, x_test, y_train, y_text = train_test_split(   # train_test_split의 파라미터 중 test_size는 전체 데이터에서 테스트 데이터 세트의 크기를 얼마로 샘플링할 것인가?
# x,y, test_size=0.3,                                     # test_size 0.3으로 설정하면 train set 0.7 
# train_size=0.7,                                         # random_state는 호출할 때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수 값.
# random_state=50)                                        # random_state는 어떤 순자를 적든 그 기능은 같다.     
                                                         # sklearn.model_selection 
print(x_train)  #[2 7 6 3 4 8 5]
print(x_test)   #[ 1  9 10]
print(y_train)
print(y_test)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x_train, y_train, epochs =100, batch_size = 1)

#4 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict([11])
print("[11의 예측 값 : ", result)

