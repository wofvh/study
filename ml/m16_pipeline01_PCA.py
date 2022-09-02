import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score, r2_score

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold , StratifiedKFold
x_train,x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True, random_state=1234)


# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


#2. 모델구성 
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA           # 300개컬럼을 한번훈련시키면 시간이 많이 걸린다. 10개로 압축해서 
from sklearn.pipeline import make_pipeline

# model = RandomForestClassifier()

# model = make_pipeline( MinMaxScaler(),StandardScaler(), RandomForestClassifier())             #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.
model = make_pipeline( MinMaxScaler(),PCA(), RandomForestClassifier())            
 #make_pipeline 은 fit할 때, 스케일러와 모델이 같이된다.
 
#3. 컴파일 훈련
import time
start_time = time.time()

model.fit(x_train,y_train)
end_time = time.time()
#4. 평가 예측
result = model.score(x_test, y_test)

print('model.score :',model.score(x_test,y_test))

y_predict= model.predict(x_test)
print('acc_score:',accuracy_score(y_test,y_predict))

print("걸린시간 :",round(end_time-start_time,4),"초")

from sklearn.metrics import accuracy_score

# nopipeline 
# model.score: 1.0
# pipeline
# model.score: 1.0



