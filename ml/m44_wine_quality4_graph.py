# 아웃라이어 확인,처리
import pandas as pd
# 11번째까지 x 나머지 y 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel   # 모델을 선택.

#1. 데이터
data = pd.read_csv('C:\study\_data\wine/winequality-white.csv',header=0,sep=';')
import matplotlib.pyplot as plt
print(data.shape)
print(data.describe()  )  

## 그래프 그려봐 ##

#1. value_counts > x
#2. groupby 써, count() 써

count_data = data.groupby('quality')['quality'].count()
print(count_data)



plt.bar(count_data.index, count_data)
plt.show()