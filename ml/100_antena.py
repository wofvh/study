import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.ensemble import BaggingClassifier,BaggingRegressor  # 한가지 모델을 여러번 돌리는 것(파라미터 조절).
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_val_score, StratifiedKFold, KFold
import tensorflow as tf
from catboost import CatBoostRegressor, Pool
path = 'D:\study_data\_data/antena/'

SEED = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED) # Seed 고정
# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
# seed_everything(42) # Seed 고정

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

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature
test_x = test_x.filter(regex = 'X')

print(train_x.shape)
print(train_y.shape)

cols = ["X_10","X_11"]
train_x[cols] = train_x[cols].replace(0, np.nan)

# MICE 결측치 보간
imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')


train_x = pd.DataFrame(imp.fit_transform(train_x))
print(train_x.shape,train_y.shape)


######################모델######################################
from sklearn.linear_model import LogisticRegression
# model = MultiOutputRegressor(RandomForestRegressor()).fit(train_x, train_y)
# 0.03932714821910016  0820_1 

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, train_y)
# 0.28798862985210744 

# model = BaggingRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=4,random_state=123) ).fit(train_x, train_y)
# 0.098387698230517  best

# model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.1, gamma = 1, subsample=1, colsample_bytree = 1, max_depth=3) ).fit(train_x, train_y)
# 0.0942562122814897

# model = XGBRegressor().fit(train_x, train_y)
# 0.4177584378415335

print('Done.')
######################모델######################################

n_splits = 5
predictions = []
lgnrmses = []
kfold = KFold(n_splits = n_splits, random_state = SEED, shuffle = True)
for i, (train_idx, val_idx) in enumerate(kfold.split(train_x)):
    preds = []
    y_vals = []
    predictions_ = []
    for j in range(1, 15):
        if j < 10:
            train_y_ = train_y[f'Y_0{j}']
        else:
            train_y_ = train_y[f'Y_{j}']
        X_train, y_train = train_x.iloc[train_idx], train_y_.iloc[train_idx]
        X_val, y_val = train_x.iloc[val_idx], train_y_.iloc[val_idx]      
        
        print(f'fit {train_y.columns[j-1]}')
        model = CatBoostRegressor(random_state = SEED)
        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0)
        
        print(f'predict {train_x.columns[j-1]}')
        pred = model.predict(X_val)
        prediction = model.predict(test_x)
        #print(prediction)
        predictions_.append(prediction)
        #print(predictions_)
        preds.append(pred)
        y_vals.append(y_val)
    predictions.append(predictions_)
    print(predictions)
    lgnrmse = lg_nrmse(np.array(y_vals).T, np.array(preds).T)
    lgnrmses.append(lgnrmse)
    print(f'Fold {i} / lg_nrmse : {lgnrmse}')
np.mean(lgnrmse)
preds = model.predict(test_x)
print(preds)
print(preds.shape)

print(model.score(train_x, train_y))
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

submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv(path + 'submmit0822_1.csv', index=False)



#0821_1 'X_04','X_23','X_47','X_48' 삭제
#0821_2 'X_07','X_08','X_-09' 삭제