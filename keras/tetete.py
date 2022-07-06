import numpy as np
print(np.__version__) # 1.20.3

# 결과확인을 위해 소수 출력 옵션 변경
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.1f}".format(x)}) 

from sklearn.preprocessing import MinMaxScaler

t1 =  np.array([
                    [ 1,    1000],
                    [ 5,   10000],
                    [10,  100000],
               ])
t2 =  np.array([
                    [  2,    100],
                    [ 15,  20000],
                    [100, 300000],
               ])

scaler = MinMaxScaler()

scaler.fit(t1)
print(scaler.n_samples_seen_, scaler.data_min_, scaler.data_max_, scaler.feature_range)
# > 3 [1.0 1000.0] [10.0 100000.0] (0, 1)

scaler.fit(t2)
print(scaler.n_samples_seen_, scaler.data_min_, scaler.data_max_, scaler.feature_range)
# > 3 [1.0 1000.0] [10.0 100000.0] (0, 1)

t2_prinme = scaler.transform(t2)


# 종류 : StandardScaler, RobustScaler, MinMaxScaler, Normalizer
