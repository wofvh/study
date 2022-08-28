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

# MICE 결측치 보간
imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')


x = pd.DataFrame(imp.fit_transform(x))
print(x.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=123)
from catboost import CatBoostRegressor  
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
n_splits = 5
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)


#############################################################################################
# cat_paramets = {"learning_rate" : (0.2,0.6),
#                 'depth' : (7,10),
#                 'od_pval' :(0.2,0.5),
#                 'model_size_reg' : (0.3,0.5),
#                 'l2_leaf_reg' :(4,8),
#                 'fold_permutation_block':(1,10),
#                 # 'leaf_estimation_iterations':(1,10)
#                 }

# def xgb_hamsu(learning_rate,depth,od_pval,model_size_reg,l2_leaf_reg,
#               fold_permutation_block,
#             #   leaf_estimation_iterations
#               ) :
#     params = {
#         'n_estimators':200,
#         "learning_rate":max(min(learning_rate,1),0),
#         'depth' : int(round(depth)),  #무조건 정수
#         'l2_leaf_reg' : int(round(l2_leaf_reg)),
#         'model_size_reg' : max(min(model_size_reg,1),0), # 0~1 사이의 값이 들어가도록 한다.
#         'od_pval' : max(min(od_pval,1),0),
#         'fold_permutation_block' : int(round(fold_permutation_block)),  #무조건 정수
#         # 'leaf_estimation_iterations' : int(round(leaf_estimation_iterations)),  #무조건 정수
#                 }
    
#     # *여러개의 인자를 받겠다.
#     # **키워드 받겠다(딕셔너리형태)
    
#     model = MultiOutputRegressor(CatBoostRegressor(**params))
    
#     model.fit(x_train,y_train,
#               verbose=0 )
#     y_predict = model.predict(x_test)
#     results = r2_score(y_test,y_predict)
    
#     return results
# xgb_bo = BayesianOptimization(f=xgb_hamsu,
#                               pbounds=cat_paramets,
#                               random_state=123)
# xgb_bo.maximize(init_points=2,
#                 n_iter=200)
# print(xgb_bo.max)

################################################################################################
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# lr = MultiOutputRegressor(CatBoostRegressor(random_state=1234,verbose=False))
# import time
# start_time = time.time()
# end_time = time.time()-start_time
# Multi_parameters= {'n_jobs':[-1]}
# cat = MultiOutputRegressor(CatBoostRegressor(random_state=123,
#                         verbose=False,
#                         learning_rate=0.2,
#                         depth= 8,
#                         od_pval =0.5,
#                         fold_permutation_block = 10,
#                         model_size_reg =0.3,
#                         l2_leaf_reg =7.246964487506609,
#                         n_estimators=500))
# model = RandomizedSearchCV(cat,Multi_parameters,cv=kfold,n_jobs=-1)


#################################################################

param_grid = [
              {'n_estimators':[10], 'max_features':[10]},
              {'bootstrap':[False],'n_estimators':[400], 'max_features':[6]}
]

forest_reg = RandomForestRegressor(n_estimators=100, random_state=2)
# 
model = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='accuracy',
                           verbose=0,
                           return_train_score=True)
######################모델######################################
from sklearn.linear_model import LogisticRegression
model = MultiOutputRegressor(RandomForestRegressor()).fit(x, y)
# 0.03932714821910016  0820_1 

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, y)
# 0.28798862985210744 

# model = BaggingRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=4,random_state=123) ).fit(train_x, y)
# 0.098387698230517  best

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=3) ).fit(x, y)
# 0.0942562122814897

# model = XGBRegressor().fit(train_x, y)
# 0.4177584378415335

print('Done.')
######################모델######################################

model.fit(x_train,y_train)  

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('r2_score:',r2)

# print(model.score(x, y))
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

for idx, col in enumerate(submit.columns):  # index값이 같이 필요할 때 enumerate를 사용 (열거하다)
                                            # 평소에는 iterater는 값만 .                (반복하다)
    if col=='ID':
        continue
    submit[col] = y_summit[:,idx-1]
print('Done.')

submit.to_csv(path + 'submmit0825_2.csv', index=False)
print('제출성공')



#0821_1 'X_04','X_23','X_47','X_48' 삭제
#0821_2 'X_07','X_08','X_-09' 삭제