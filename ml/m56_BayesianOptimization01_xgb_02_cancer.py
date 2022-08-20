
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from xgboost import XGBClassifier
#1. 데이터

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=123,shuffle=True)

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

# parameters = {'n_estimators':[100, 200, 300, 400, 500, 1000],
#               'learning_rate':[0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
#               'max_depth':[None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#               'gamma': [0, 1, 2, 3, 4, 5, 7, 10 ,100] ,
#               'min_child_weight':[0, 1, 2, 3, 4, 5, 7, 10 ,100] ,
#               'subsample':[0, 1, 2, 3, 4, 5, 7, 10 ,100] ,
#             #   'colsample_bytree':[0, 1, 2, 3, 4, 5, 7, 10 ,100] ,
#               'colsample_bylevel':[0,0.1,0.2,0.3,0.5,0.7,1],
#             #   'colsample_byload':[1],
#               'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
#               'reg_lambda':[0,0.1,0.01,0.001,1,2,10]
#               }  

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
    
    model = XGBClassifier(**params)
    # ** 키워드받겠다(딕셔너리형태)
    # * 여러개의인자를 받겠다.
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50,
              )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)
    
    
    return  results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds= xgb_params,
                              random_state=123)
lgb_bo.maximize(init_points=5,n_iter=50)

print(lgb_bo.max)

# {'target': 0.9912280701754386,
#  'params': {'colsample_bytree': 0.7919410497955786,
#             'max_bin': 458.2130768347505,
#             'max_depth': 13.011875840403292,
#             'min_child_sample': 90.11618557959497,
#             'min_child_weight': 4.849037464942212,
#             'num_leaves': 42.53863680221558,
#             'reg_alpha': 30.319611432881363,
#             'reg_lambda': 4.135217639511628,
#             'subsample': 0.5181827326249917}}

