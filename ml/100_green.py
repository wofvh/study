 
from audioop import lin2ulaw
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split













# path = './_data/green/train_target/'
# dd = pd.read_csv(path + 'CASE_01')

# path = './_data/green/'
# test_input = path + 'test_input/'
# test_target = path + 'test_target/'
# train_input = path + 'train_input/'
# train_target = path + 'train_target/'

# x_train = pd.read_csv(test_input+ 'TEST_01')

# path = 'D:\study_data\_data\green/'
# train_input_path = path+'train_input/'
# test_input_path = path+'test_input/'
# train_target_path = path+'train_target/'
# test_target_path = path+'test_target/'

# train_input = pd.read_csv(train_input_path+'CASE_01.csv')
# test_input = pd.read_csv(test_input_path+'TEST_01.csv')
# train_target = pd.read_csv(train_target_path+'CASE_01.csv')
# test_target = pd.read_csv(test_target_path+'TEST_01.csv')
# print(train_input)

import pandas as pd
import numpy as np
import glob

path = 'D:\study_data\_data\green/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

x_train,y_train = aaa(train_input_list, train_target_list) #, False)
x_test,y_test = aaa(val_input_list, val_target_list) #, False)

# print(train_data[0])
# print(len(train_data), len(label_data)) # 1607 1607
# print(len(train_data[0]))   # 1440
# print(label_data)   # 1440
# print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)

print(x_train.shape)        # (1607, 1440, 37)
print(y_train.shape)        # (1607,)
print(x_test.shape)         # (206, 1440, 37)
print(y_test.shape)         # (206,)

from sklearn.metrics import r2_score, accuracy_score
# x_train.reshape(1607*1440)


# x_train, x_test, y_train, y_test = train_test_split(train_data.shape,label_data.shape,
#                                                     train_size=0.8,
#                                                     random_state=66
#                                                     )
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.linear_model import LogisticRegression, LinearRegression     # LogisticRegression 분류모델 LinearRegression 회귀
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBClassifier, XGBRegressor

###################################################
# model = Sequential()
# model.add(Conv1D(64,2 ,input_shape=(1440,37)))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(1))
####################################################
model=XGBClassifier()

model.fit(x_train, y_train)

#4. 평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)
# y_predict = model.predict(x_test)
# from sklearn.metrics import r2_score, accuracy_score
# r2 = r2_score(y_test, y_predict)
# print('acc :', r2)
#####################################################

from sklearn.metrics import accuracy_score, r2_score

result = model.score(x_test,y_test)
print("model.score:",result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)

print( 'accuracy_score :',acc)
print(model,':')   
print("===================================")




























































exit()
path = './_data/green/train_target/'
file_list = os.listdir(path)
# print(file_list)

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    
train_target = pd.concat(li, axis=0, ignore_index=True)
##############################################################
 
path = './_data/green/train_input/'
file_list = os.listdir(path)
print(file_list)

all_files = glob.glob(path + "/*.csv")

li2 = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li2.append(df)

train_input = pd.concat(li2, axis=0, ignore_index=True)
##############################################################

path = './_data/green/test_target/'
file_list = os.listdir(path)
print(file_list)

all_files = glob.glob(path + "/*.csv")

li3 = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li3.append(df)

test_target = pd.concat(li3, axis=0, ignore_index=True)
##############################################################

path = './_data/green/test_input/'
file_list = os.listdir(path)
print(file_list)

all_files = glob.glob(path + "/*.csv")

li4 = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li4.append(df)

test_input = pd.concat(li4, axis=0, ignore_index=True)
##############################################################

print(train_target.shape) # y_train
print(train_input.shape)  # x_train
print(test_target.shape)  # y_test
print(test_input.shape)   # x_train

(1813, 2)
(2611507, 43)
(195, 2)
(285120, 42)






