import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8,np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan,4,np.nan,8,np.nan]])

# print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)


print(data.isnull())
print(data.isnull().sum())
print(data.info())

#1.결측지 삭제 

print('-------------결측지 삭제------------------')
print(data.dropna())
print(data.dropna(axis=0))
print(data.dropna(axis=1))

# 2-1 특정값- 평균
print('-------------결측지 처리 mean()------------------') # 컬럼별 평균을 구하기 때문에 문제가 있다. 
means = data.mean()
print("평균:",means)
data2 =data.fillna(means)
print(data2)

# 2-2 특정값- 중위
print('-------------결측지 처리 median()------------------') # 컬럼별 평균을 구하기 때문에 문제가 있다. 
median = data.median()
print("평균:",median)
data3 =data.fillna(median)
print(data3)
# exit()
# ts = pd.Series([2,np.nan,np.nan,8,10],index=dates)

# print(ts)
# print("----------------------------------")
# ts = ts.interpolate()