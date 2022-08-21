
from bayes_opt import BayesianOptimization, bayesian_optimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from xgboost import XGBClassifier, XGBRFRegressor
#1. 데이터

datasets =load_diabetes()

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
    
    model =LGBMRegressor(**params)
    # ** 키워드받겠다(딕셔너리형태)
    # * 여러개의인자를 받겠다.
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='error',
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

# print(lgb_bo.max)

exit()
##################최적의 파라미터 ##########################3
model = XGBRFRegressor(colsample_bytree =0.66596919092928,
                      max_depth= 7, 
                      min_child_sample= 86.46653678311762,
                      min_child_weight=  8.180087453093222,
                      reg_alpha=  3.540066324626541,
                      reg_lambda= 12.844583766937252,
                      subsample=  0.6508162099908998)
##################최적의 파라미터 ##########################3
# model = XGBClassifier()

model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

print(model.score(x_test, y_test))  

acc= accuracy_score(y_test,y_predict)

print('acc :',acc)

################## 그냥 ############################
0.9666666666666667
acc : 0.9666666666666667
##################최적의 파라미터 ##########################3
# 1.0
# acc : 1.0



