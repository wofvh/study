from pkgutil import ImpImporter
import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8,np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan,4,np.nan,8,np.nan]])

# print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
# print(data)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# imputer = SimpleImputer()
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent')               # 가장 빈번한 값을 채워 넣겠다. 개수가 똑같을 시 가장 앞에있는 값으로 채워넣음.
# imputer = SimpleImputer(strategy='constant')                    # constant = 디폴트 0
# imputer = SimpleImputer(strategy='constant',fill_value=777) 
  
# imputer =  KNNImputer()           #  디폴트 = mean 
# 마찬가지로 Scikit learn의 KNNImputer 함수를 사용한다. 평균, 중앙값, 최빈값으로 대치하는 경우보다 더 정확할 때가 많다. 
# 반면 전체 데이터셋을 메모리에 올려야 해서 메모리가 많이 필요하고 이상치에 민감하다는 단점이 있다. 
# 대치값을 판단할 기준이 되는 이웃의 개수는 n_neighbors 파라미터로 설정할 수 있다.

imputer =  IterativeImputer()      # 다변량 대치 방법                                
# 다른 모든 특성에서 각 특성을 추정하는 다변량 대치방법이다. Scikit Learn의 IterativeImputer 함수를 이용한다. 
# 해당 함수를 사용하기 위해서는 enable_iterative_imputer의 import가 선행되어야 한다. 라운드 로빈 방식으로 다른 
# feature들로 결측값이 있는 feature를 모델링 하여 결측값을 대치한다.


imputer.fit(data)
data2 = imputer.transform(data) 
print(data2)
