from time import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# 데이터 가져오기 
path = 'C:\study\_data/travel/' 
train = pd.read_csv(path + 'train.csv',index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)

print(train.describe()) 
print(test.describe()) 

# 결측지 확인
print(train.isnull().sum())
# Age                          94
# TypeofContact                10
# CityTier                      0
# DurationOfPitch             102
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            13
# ProductPitched                0
# PreferredPropertyStar        10
# MaritalStatus                 0
# NumberOfTrips                57
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     27
# Designation                   0
# MonthlyIncome               100
# ProdTaken                     0
print(test.isnull().sum())
# Age                         132
# TypeofContact                15
# CityTier                      0
# DurationOfPitch             149
# Occupation                    0
# Gender                        0
# NumberOfPersonVisiting        0
# NumberOfFollowups            32
# ProductPitched                0
# PreferredPropertyStar        16
# MaritalStatus                 0
# NumberOfTrips                83
# Passport                      0
# PitchSatisfactionScore        0
# OwnCar                        0
# NumberOfChildrenVisiting     39
# Designation                   0
# MonthlyIncome               133


#전처리
train['Age'].fillna(train.groupby('Designation')['Age'].transform('mean'), inplace=True)
test['Age'].fillna(test.groupby('Designation')['Age'].transform('mean'), inplace=True)

train['Age']=np.round(train['Age'],0).astype(int)
test['Age']=np.round(test['Age'],0).astype(int)

combine = [train,test]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5
    
train['TypeofContact'].fillna('Self Enquiry', inplace=True)
test['TypeofContact'].fillna('Self Enquiry', inplace=True)

train['DurationOfPitch']=train['DurationOfPitch'].fillna(0)
test['DurationOfPitch']=test['DurationOfPitch'].fillna(0)

train['MonthlyIncome'].fillna(train.groupby('Designation')['MonthlyIncome'].transform('median'), inplace=True)
test['MonthlyIncome'].fillna(test.groupby('Designation')['MonthlyIncome'].transform('median'), inplace=True)

train['NumberOfChildrenVisiting'].fillna(train.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('median'), inplace=True)
test['NumberOfChildrenVisiting'].fillna(test.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('median'), inplace=True)

train['NumberOfFollowups'].fillna(train.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('median'), inplace=True)
test['NumberOfFollowups'].fillna(test.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('median'), inplace=True)

train['PreferredPropertyStar'].fillna(train.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test['PreferredPropertyStar'].fillna(test.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

train['NumberOfTrips'].fillna(train.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test['NumberOfTrips'].fillna(test.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)

train.loc[ train['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test.loc[ test['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train.loc[ train['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test.loc[ test['Gender'] =='Fe Male' , 'Gender'] = 'Female'



# 이상치 처리
cols = ['TypeofContact','ProductPitched','Designation','MaritalStatus','Occupation','Gender']
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train[col]=le.fit_transform(train[col])
    test[col]=le.fit_transform(test[col])

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) 
    print("1사분위 : ",quartile_1) 
    print("q2 : ",q2)  
    print("3사분위 : ",quartile_3)
    iqr =quartile_3-quartile_1  
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
                     
DurationOfPitch_out_index= outliers(train['DurationOfPitch'])[0] #44
Gender_out_index= outliers(train['Gender'])[0]
NumberOfPersonVisiting_out_index= outliers(train['NumberOfPersonVisiting'])[0] # 1
NumberOfFollowups_out_index= outliers(train['NumberOfFollowups'])[0]
ProductPitched_index= outliers(train['ProductPitched'])[0]
PreferredPropertyStar_out_index= outliers(train['PreferredPropertyStar'])[0] 
MaritalStatus_out_index= outliers(train['MaritalStatus'])[0]
NumberOfTrips_out_index= outliers(train['NumberOfTrips'])[0] # 38
Passport_out_index= outliers(train['Passport'])[0]
PitchSatisfactionScore_out_index= outliers(train['PitchSatisfactionScore'])[0]
OwnCar_out_index= outliers(train['OwnCar'])[0]
NumberOfChildrenVisiting_out_index= outliers(train['NumberOfChildrenVisiting'])[0] 
Designation_out_index= outliers(train['Designation'])[0] 
MonthlyIncome_out_index= outliers(train['MonthlyIncome'])[0] 
'''
lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
                              
print(len(lead_outlier_index)) #577

lead_not_outlier_index = []
for i in train.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_clean = train.loc[lead_not_outlier_index]      
train_clean = train_clean.reset_index(drop=True)
# print(train_clean)
'''  
# 필요없는 컬럼 제거     
x = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting',
                'OwnCar', 'MonthlyIncome', 'NumberOfFollowups',], axis=1)
# x = train.drop(['ProdTaken'], axis=1)
test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting',
                'OwnCar', 'MonthlyIncome', 'NumberOfFollowups',], axis=1)
y = train['ProdTaken']

# train, test 분리
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)


# 모델

n_splits = 6
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=1234)

cat_paramets = {"learning_rate" : [0.01],
                'depth' : [8],
                'od_pval' : [0.12673190617341812],
                # 'model_size_reg': [0.44979263197508923],
                'fold_permutation_block': [142],
                'l2_leaf_reg' :[0.33021257848638497]}
cat = CatBoostClassifier(random_state=72,verbose=False,n_estimators=1304)
model = RandomizedSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1,)

# import time 
# start_time = time.time()
model.fit(x_train,y_train)   
# end_time = time.time()-start_time 
y_predict = model.predict(x_test)
results = accuracy_score(y_test,y_predict)
print('매개변수 : ',model.best_params_)
print('점수 : ',model.best_score_)
print('acc :',results)
# print('걸린 시간 :',end_time)

# 훈련
# model.fit(x,y)

# 제출
y_submmit = model.predict(test)
y_submmit = np.round(y_submmit,0)
submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
                      )
submission['ProdTaken'] = y_submmit

submission.to_csv('test0902_3.csv',index=False)
print('완료')

##########
# 최상의 점수 :  0.8930338463986
# acc : 0.9418604651162791
# 걸린 시간 : 11.291642665863037

############ RandomState = 100
# 최상의 점수 :  0.8813139873889755
# acc : 0.921875

