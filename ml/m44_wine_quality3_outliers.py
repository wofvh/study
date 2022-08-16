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

print(data.shape)
print(data.describe()  )            # pandas에서 좋은점 자료정보가 다나옴 

#     fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol      quality
# count    4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000  4898.000000  4898.000000  4898.000000  4898.000000
# mean        6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027     3.188267     0.489847    10.514267     5.877909
# std         0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991     0.151001     0.114126     1.230621     0.885639
# min         3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110     2.720000     0.220000     8.000000     3.000000
# 25%         6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723     3.090000     0.410000     9.500000     5.000000
# 50%         6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740     3.180000     0.470000    10.400000     6.000000
# 75%         7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100     3.280000     0.550000    11.400000     6.000000
# max        14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980     3.820000     1.080000    14.200000     9.000000

print(data.info())
#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         4898 non-null   float64
#  1   volatile acidity      4898 non-null   float64
#  2   citric acid           4898 non-null   float64
#  3   residual sugar        4898 non-null   float64
#  4   chlorides             4898 non-null   float64
#  5   free sulfur dioxide   4898 non-null   float64
#  6   total sulfur dioxide  4898 non-null   float64
#  7   density               4898 non-null   float64
#  8   pH                    4898 non-null   float64
#  9   sulphates             4898 non-null   float64
#  10  alcohol               4898 non-null   float64
#  11  quality               4898 non-null   int64      #분류모델은 int64 형태가 정상.
# dtypes: float64(11), int64(1)
# memory usage: 459.3 KB
# None

data = data.to_numpy()
print(type(data))

# data= data.values
# x = np.array(data.drop['quality'])
# y = np.array(data['quality'])
def outliers(data_out, i) :
    quartile_1, q2, quartile_3 = np.percentile(data_out[:,i],
                                               [25,50,75])
    print('1사분위 :',quartile_1)               # 1사분위 : 4.0
    print('q2:',q2)                             # q2: 7.0  (중위값)
    print('3사분위 :',quartile_3)               # 3사분위 : 10.0
    iqr = quartile_3 - quartile_1
    print('iqr:', iqr)                          # iqr: 6.0
    lower_bound = quartile_1 -(iqr *1.5)
    upper_bound = quartile_3 +(iqr *1.5)
    print('upper_bound:',upper_bound)           # upper_bound: 19.0
    print('lower_bound:',lower_bound)           # lower_bound: -5.0
    return np.where((data_out[:,i]>upper_bound)|     # aaa에서 upper_bound(19.0)보다 높은 번째 수 8
                    (data_out[:,i]<lower_bound)) 
    
outliers_loc0 = outliers(data,0) #outliers(data_out, i)
outliers_loc1 = outliers(data,1)
outliers_loc2 = outliers(data,2)
outliers_loc3 = outliers(data,3)
outliers_loc4 = outliers(data,4)
outliers_loc5 = outliers(data,5)
outliers_loc6 = outliers(data,6)

print('이상치[1]의 위치:',outliers_loc0)
print('이상치[2]의 위치:',outliers_loc1)            # 이상치의 위치: (array([0, 12], dtype=int64),
print('이상치[3]의 위치:',outliers_loc2)            # 이상치의 위치: (array([0, 12], dtype=int64),
print('이상치[4]의 위치:',outliers_loc3)            # 이상치의 위치: (array([0, 12], dtype=int64),
print('이상치[5]의 위치:',outliers_loc4)            # 이상치의 위치: (array([0, 12], dtype=int64),
print('이상치[6]의 위치:',outliers_loc5)            # 이상치의 위치: (array([0, 12], dtype=int64),
print('이상치[7]의 위치:',outliers_loc6)            # 이상치의 위치: (array([0, 12], dtype=int64),
    
import matplotlib.pyplot as plt
plt.boxplot(data)
# plt.show() 
print(data[0])  
exit()
data = data[0].delete([  98,  169,  207,  294,  358,  551,  555,  656,  774,  847,  873,
       1053, 1109, 1123, 1124, 1138, 1139, 1141, 1142, 1146, 1147, 1178,
       1205, 1210, 1214, 1228, 1239, 1263, 1300, 1307, 1308, 1309, 1312,
       1313, 1334, 1349, 1372, 1373, 1404, 1420, 1423, 1505, 1526, 1536,
       1544, 1561, 1564, 1580, 1581, 1586, 1621, 1624, 1626, 1627, 1690,
       1718, 1730, 1758, 1790, 1801, 1856, 1857, 1858, 1900, 1930, 1932,
       1936, 1951, 1961, 2014, 2017, 2028, 2030, 2050, 2083, 2127, 2154,
       2162, 2191, 2206, 2250, 2266, 2308, 2312, 2321, 2357, 2378, 2400,
       2401, 2404, 2535, 2540, 2541, 2542, 2607, 2625, 2639, 2668, 2872,
       3094, 3095, 3220, 3265, 3307, 3410, 3414, 3526, 3710, 3915, 4259,
       4446, 4470, 4518, 4522, 4679, 4786, 4787, 4792, 4847]) 
    
  
exit()
x = data.values[:,0:11]
y = data.values[:,11]

print(x.shape,y.shape)
print(np.unique(y, return_counts= True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# 다중분류에서는 분포도 반드시 확인.

print(data['quality'].value_counts())  # 분포도 확인 
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    shuffle=True,
                                                    stratify=y)

from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가
y_predict = model.predict(x_test)

score = model.score(x_test,y_test)
print('결과:',score)
print('acc:', accuracy_score(y_test,y_predict))
print('f1_macro:',f1_score(y_test,y_predict, average='macro'))      # 이진분류 f1스코어 프리시즌(재현) 리콜(정밀도) 
print('f1_micro:',f1_score(y_test,y_predict, average='micro'))      # acc 와 동일. https://m.blog.naver.com/sw4r/221681933731(과제)

# 결과: 0.7285714285714285
# acc: 0.7285714285714285
# f1_macro: 0.43722325433133896
# f1_micro: 0.7285714285714285