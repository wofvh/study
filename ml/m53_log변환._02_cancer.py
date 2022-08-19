from cgi import test
from sklearn.datasets import load_boston,load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt

datasets =  load_breast_cancer()
x,y = datasets.data, datasets.target
# print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234,
)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = LogisticRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = accuracy_score(y_test, y_predict)
print("그냥 결과 : ", round(results, 4))


############################ 로그 변환 ############################ 
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)

df.plot.box()
plt.title('boston')
plt.xlabel('Feature')
plt.ylabel('데이터값')

plt.show()

# print(df['concave'].head())                 #  그냥 결과 :  0.7665
df['dimension'] = np.log1p(df['dimension'])           #  그냥 결과 :  0.7711
# print(df['concave'].head())

exit()

# df['CRIM'] = np.log1p(df['CRIM'])   # 로그변환 결과 :  0.7596
# df['ZN'] = np.log1p(df['ZN'])       # 로그변환 결과 :  0.7734
# df['TAX'] = np.log1p(df['TAX'])     # 로그변환 결과 :  0.7669
                                    # 3개 모두 쓰면 : 0.7785
                                    
x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=1234,
)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
model = LogisticRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = accuracy_score(y_test, y_predict)
print("로그변환 결과 : ", round(results, 4))

# 'sepal width (cm)'제거
# 그냥 결과 :  1.0
# 로그변환 결과 :  1.0