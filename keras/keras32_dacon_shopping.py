import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from sqlalchemy import true                                 #pandas : 엑셀땡겨올때 씀
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

train = pd.read_csv('./_data/shopping/train.csv')

test = pd.read_csv('_data/shopping/test.csv')

sample_submission = pd.read_csv('_data/shopping/sample_submission.csv')


print(train)  # [6255 rows x 13 columns]
print(test)   # [180 rows x 12 columns]

#1. 데이터
path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv',                 # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0)                        # index_col=n n번째 컬럼을 인덱스로 인식
print(train_set)
print(train_set.shape) # (6255, 12)

test_set = pd.read_csv(path + 'test.csv',                    # 예측에서 쓸거임                
                       index_col=0)
print(test_set)
print(test_set.shape) # (180, 11)

print(train_set.columns)

# ['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1',
#       'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment',
#       'IsHoliday', 'Weekly_Sales']
print(train_set.shape)   # (6255, 12)

print(train_set.info())  

# Int64Index: 6255 entries, 1 to 6255
# Data columns (total 12 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   Store         6255 non-null   int64
#  1   Date          6255 non-null   object
#  2   Temperature   6255 non-null   float64
#  3   Fuel_Price    6255 non-null   float64
#  4   Promotion1    2102 non-null   float64
#  5   Promotion2    1592 non-null   float64
#  6   Promotion3    1885 non-null   float64
#  7   Promotion4    1819 non-null   float64
#  8   Promotion5    2115 non-null   float64
#  9   Unemployment  6255 non-null   float64
#  10  IsHoliday     6255 non-null   bool
#  11  Weekly_Sales  6255 non-null   float64
# dtypes: bool(1), float64(9), int64(1), object(1)     

print(train_set.describe())  
#              Store  Temperature  ...  Unemployment  Weekly_Sales
# count  6255.000000  6255.000000  ...   6255.000000  6.255000e+03 
# mean     23.000000    60.639199  ...      8.029236  1.047619e+06 
# std      12.988211    18.624094  ...      1.874875  5.654362e+05 
# min       1.000000    -2.060000  ...      4.077000  2.099862e+05 
# 25%      12.000000    47.170000  ...      6.916500  5.538695e+05 
# 50%      23.000000    62.720000  ...      7.906000  9.604761e+05 
# 75%      34.000000    75.220000  ...      8.622000  1.421209e+06 
# max      45.000000   100.140000  ...     14.313000  3.818686e+06 

# [8 rows x 10 columns]

print(train_set.isnull().sum())

# Store              0
# Date               0
# Temperature        0
# Fuel_Price         0
# Promotion1      4153
# Promotion2      4663
# Promotion3      4370
# Promotion4      4436
# Promotion5      4140
# Unemployment       0
# IsHoliday          0
# Weekly_Sales       0
# dtype: int64

train_set = train_set.fillna(0)       # dropna() : train_set 에서 na, null 값 들어간 행 삭제
test_set = test_set.fillna(0) # test_set 에서 이빨빠진데 바로  ffill : 위에서 가져오기 test_set.mean : 평균값
print(train_set.isnull().sum()) 
print(train_set.shape)   # (6255, 12)

x = train_set.drop(['Weekly_Sales'], axis=1)                    # drop 데이터에서 ''사이 값 빼기
print(x)
#       Store        Date  ...  Unemployment  IsHoliday
# id                       ...
# 1         1  05/02/2010  ...         8.106      False
# 2         1  12/02/2010  ...         8.106       True
# 3         1  19/02/2010  ...         8.106      False
# 4         1  26/02/2010  ...         8.106      False
# 5         1  05/03/2010  ...         8.106      False
# ...     ...         ...  ...           ...        ...
# 6251     45  31/08/2012  ...         8.684      False
# 6252     45  07/09/2012  ...         8.684       True
# 6253     45  14/09/2012  ...         8.684      False
# 6254     45  21/09/2012  ...         8.684      False
# 6255     45  28/09/2012  ...         8.684      False

# [6255 rows x 11 columns]
print(x.columns)
print(x.shape)  # (6255, 11)

y = train_set['Weekly_Sales'] 
print(y)
print(y.shape)  # (6255,)

# cols 



