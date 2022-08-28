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

print('-------------결측지 삭제(시작)-------------------------')
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
print('-------------결측지 처리 median()----------------') # 중위값
median = data.median()
print("평균:",median)
data3 =data.fillna(median)
print(data3)
exit()
#2-2 특정값 - ffill,bill
print('-------------결측지 처리ffill----------------') # 앞에 값으로 채운다. 
data4 = data.fillna(method='ffill')
print('ffill:',data4)

#2-3 특정값 - ffill,bill
print('-------------결측지 처리bfill----------------') # 뒤에 값으로 채운다. 
data5 = data.fillna(method='bfill')
print('ffill:',data5)

#2-4 특정값 - 임의의값으로 채우기 
print('-------------결측지 임의의값----------------') #임의의값에 값으로 채운다. 
data6 = data.fillna(value=77777)
print('ffill:',data6)

#2-5 특정컬럼만 
print('-------------특정컬럼----------------') #특정컬럼. 
means = data['x1'].mean()
print(means)
data['x1'] = data['x1'].fillna(means)
print(data)

meds = data['x2'].median()
print(meds)
data['x2'] = data['x2'].fillna(meds)
print(data)

data['x4'] = data['x4'].fillna(77777)
print(data)




# exit()
# ts = pd.Series([2,np.nan,np.nan,8,10],index=dates)

# print(ts)
# print("----------------------------------")
# ts = ts.interpolate()