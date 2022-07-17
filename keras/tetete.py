import numpy as np
x = np.array([0.1,0.7,0.3,0.5,0.4])                   #데이터변경
y = np.array([1,2,3,4,5])

x =np.around(x,0)                  # x를 np.aroud를 사용해서 x가 0.5 이상이면 1 이하이면 0으로 변경 
print(x)
x =np.where(x >0.5,1,0)            # x를 np.where를 사용해서 x가 0.5 보다 크면 1(맞는값), 0(틀린값)



# print(np.__version__) # 1.20.3

# # 결과확인을 위해 소수 출력 옵션 변경
# np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.1f}".format(x)}) 

# from sklearn.preprocessing import MinMaxScaler

# t1 =  np.array([
#                     [ 1,    1000],
#                     [ 5,   10000],
#                     [10,  100000],
#                ])
# t2 =  np.array([
#                     [  2,    100],
#                     [ 15,  20000],
#                     [100, 300000],
#                ])

# scaler = MinMaxScaler()

# scaler.fit(t1)
# print(scaler.n_samples_seen_, scaler.data_min_, scaler.data_max_, scaler.feature_range)
# # > 3 [1.0 1000.0] [10.0 100000.0] (0, 1)

# scaler.fit(t2)
# print(scaler.n_samples_seen_, scaler.data_min_, scaler.data_max_, scaler.feature_range)
# # > 3 [1.0 1000.0] [10.0 100000.0] (0, 1)

# t2_prinme = scaler.transform(t2)


# # 종류 : StandardScaler, RobustScaler, MinMaxScaler, Normalizer

df['Date Time'] = pd.to_datetime(df['Date Time'])

df['year'] = df['Date Time'].dt.strftime('%Y')                  # 0000자리면 대문자, 00자리면 소문자  
df['month'] = df['Date Time'].dt.strftime('%m')              
df['day'] = df['Date Time'].dt.strftime('%d')      
df['hour'] = df['Date Time'].dt.strftime('%h')      
df['minute'] = df['Date Time'].dt.strftime('%M')      


df = df.drop(['Date Time'], axis=1)  

cols = ['year','month','day','hour','minute']
for col in cols:
    le = LabelEncoder()
    df[col]=le.fit_transform(df[col])