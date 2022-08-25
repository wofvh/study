import pandas as pd
import random
import os
import numpy as np
from sklearn import model_selection, preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.ensemble import BaggingClassifier,BaggingRegressor  # 한가지 모델을 여러번 돌리는 것(파라미터 조절).
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold, KFold
import tensorflow as tf
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
path = 'D:\study_data\_data/antena/'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

train_df = pd.read_csv(path + 'train.csv')
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
train = np.array(train_df)

print("=============================상관계수 히트 맵==============")
print(train_df.corr())                    # 상관관계를 확인.  
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=0.3)
sns.heatmap(data=train_df.corr(),square=True, annot=True, cbar=True) 
# plt.show()
# # 4,23,47,48

precent = [0.20,0.40,0.60,0.80]


print(train_df.describe(percentiles=precent))
# print(train_df.info())  
# print(train_df.columns.values)
# print(train_df.isnull().sum())

#  X_07, X_08, X_09
def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score

x = train_df.filter(regex='X') # Input : X Featrue
y = train_df.filter(regex='Y') # Output : Y Feature



print(x.shape)
print(y.shape)

cols = ["X_10","X_11"]
x[cols] = x[cols].replace(0, np.nan)

# # MICE 결측치 보간
# imp = IterativeImputer(estimator = LinearRegression(), 
#                        tol= 1e-10, 
#                        max_iter=30, 
#                        verbose=2, 
#                        imputation_order='roman')


# x = pd.DataFrame(imp.fit_transform(x))
# print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=123)
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xgb_params = {
  
    'max_depth':(1,10),
    'min_child_sample' :(0,200),
    'min_child_weight' : (0,200),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'reg_lambda' : (0.0001,100),
    'reg_alpha':(0.01,100)
}

def lgb_hamsu(max_depth,min_child_sample,min_child_weight,subsample,colsample_bytree,
              reg_lambda,reg_alpha) :
    params ={ 
             'n_estimators':500,"learning_rate":0.02,
             'max_depth': int(round(max_depth)),                 # 정수만 
            #  'num_leaves':int(round(num_leaves)),                # 정수만
             'min_child_sample' :int(round(min_child_sample)),   # 정수만
             'min_child_weight' : int(round(min_child_weight)),  # 정수만
             'subsample':max(min(subsample,1),0,),               # (0~1)사이값
             'colsample_bytree':max(min(colsample_bytree,1),0,), # (0~1)사이값
            #  'max_bin' : max(int(round(max_bin)),10),            # 10 이상
             'reg_lambda' : max(reg_lambda,0),                   # 0이상(양수)
             'reg_alpha':max(reg_alpha,0)                        # 0이상(양수)
             
            }
    
    model =MultiOutputRegressor(XGBRegressor(**params))
    # ** 키워드받겠다(딕셔너리형태)
    # * 여러개의인자를 받겠다.
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              # eval_metric='rmse',
              verbose=0,
            #   early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)
    
    
    return  results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds= xgb_params,
                              random_state=123)
lgb_bo.maximize(init_points=5,n_iter=50)

print(lgb_bo.max)

######################모델######################################
from sklearn.linear_model import LogisticRegression
# model = MultiOutputRegressor(RandomForestRegressor()).fit(x, y)
# 0.03932714821910016  0820_1 

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, y)
# 0.28798862985210744 

# model = BaggingRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=4,random_state=123) ).fit(train_x, y)
# 0.098387698230517  best

model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=3) ).fit(x, y)
# 0.0942562122814897

# model = XGBRegressor().fit(train_x, y)
# 0.4177584378415335

print('Done.')
######################모델######################################

model.fit(x_train,y_train)  

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score:',r2)

print(model.score(x, y))
print('Done.')

# {'n_estimators':[1000],
#               'learning_rate':[0.1],
#               'max_depth':[3],
#               'gamma': [1],
#               'min_child_weight':[1],
#               'subsample':[1],
#               'colsample_bytree':[1],
#               'colsample_bylevel':[1],
#             #   'colsample_byload':[1],
#               'reg_alpha':[0],
#               'reg_lambda':[1]
#               }  

####################제출############################
y_summit = model.predict(test_x)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                     )
submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'submmit0822_1.csv', index=False)
print('제출성공')



#0821_1 'X_04','X_23','X_47','X_48' 삭제
#0821_2 'X_07','X_08','X_-09' 삭제