
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from sqlalchemy import true
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y))

#겟더미
y = pd.get_dummies(y)
print(y)

#원핫인코더
df = pd.DataFrame(y)
print(df)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y = oh.fit_transform(df)
print(y)


# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
#       dtype=int64))
#######################################################################

# datasets = fetch_covtype()
# x = datasets.data
# y = datasets.target

# print (x.shape, y.shape)    # (581012 ,54)
# print ( np.unique(y,return_counts=True))      
# # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], 
# #  dtype=int64))
    
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=58525
                                                    )


#2. 모델구성
model = Sequential()
model.add(Dense(54, input_dim=54, activation='relu'))  #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dense(100, activation='relu'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
model.add(Dense(80, activation='relu'))                # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
model.add(Dense(15, activation='relu'))               
model.add(Dense(7, activation='softmax'))              # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
                                                       # ex) 70, 20, 10 -> 0.7, 0.2, 0.1

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=350, mode='auto', verbose=1, 
                              restore_best_weights=True)   


model.fit(x_train, y_train, epochs=200, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

# results= model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])

y_predict = model.predict(x_test)

#print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
# print(y_predict)
y_test = np.argmax(y_test, axis= 1)


acc= accuracy_score(y_test, y_predict) 
print('acc : ', acc) 

# print(y_predict)
print('loss : ', loss[0])
#loss식의 첫번째
print('acc :',  loss[1])
#loss식의 두번째
print('acc', acc)
# 과제 

# acc :  0.4877929141244202
# loss :  1.2024099826812744

# 3가지 원핫인코딩 방식을 비교할것 
# 1. pandas의 get_dummies

# 2. tensorflow의 to_categorical
# to_categorical은 0부터 시작하기 때문에 8개로 인식한다. 

# 3. sklearn의 OneHotEncoder
# 0이 없으면 1부터 카운트한다. 

# 미세한 차이를 정리하시오. 