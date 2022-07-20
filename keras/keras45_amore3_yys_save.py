import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Reshape,LSTM,Conv1D,Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 

#1. 데이터
path = './_data/test_amore_0718/'
    
# am = pd.read_csv( path + '삼성전자220718.csv', encoding='CP949')
# ss = pd.read_csv( path + '아모레220718.csv', encoding='CP949')
am = pd.read_csv( path + '아모레220718.csv',encoding='CP949',thousands=',')
ss = pd.read_csv( path + '삼성전자220718.csv', encoding='CP949',thousands=',')


am.at[1035:,'종가'] = 0
print(am) #2018/05/04

ss.at[1035:,'종가'] = 0
# print(ss) #2018/05/04

ss = ss[ss['종가'] > 100] #[1035 rows x 17 columns]
print(ss.shape)
print(ss)
am = am[am['종가'] > 100] #[1035 rows x 17 columns]
print(am.shape,ss.shape)
print(am) #2018/05/04
print(am,ss)      # (3040, 17) (3180, 17)  

am = am.sort_values(by='일자',ascending = False)
ss = ss.sort_values(by='일자',ascending = False)

print(am,ss)

y = np.array(am['종가'])   
print(y.shape)   # [1035 rows x 17 columns]  (1035,)

# am = am.sort_index(ascending=False)
# print(am)
 

# y = y.drop([1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783],axis=0)
# am = am.drop([1773,1774,1775,1776,1777,1778,1779,1780,1781,1782,1783],axis=0)
# ss = ss.drop([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
#               28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
#               54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
#               80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
#               101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,
#               119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,
#               137,138,139,140,141,142,143,144,145,146,147,148,1037,1038,1039],axis=0)

print(am.shape,ss.shape)      # (3029, 17) (3177, 17)
#############################################################################

am = am.rename(columns={'Unnamed: 6':'증감량'})
ss = ss.rename(columns={'Unnamed: 6':'증감량'})
print(am.info())
print(ss.info())  


am = am.drop(columns=['전일비','증감량','금액(백만)','개인','외인(수량)','외국계','프로그램','외인비'])
ss = ss.drop(columns=['전일비','증감량','금액(백만)','개인','외인(수량)','외국계','프로그램','외인비'])
aa = am['종가']
# '금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'
# am1 = am.drop(['전일비'], axis = 1) 
# ss1 = ss.drop(['전일비'], axis = 1)

am.info()
# am1 = am.drop(['전일비'], axis = 1)
# am1 = am.drop(['전일비'], axis = 1)
# test_set = test_set.drop(['Name'], axis = 1)

# train_set = train_set.drop(['Ticket'], axis = 1)
# test_set = test_set.drop(['Ticket'], axis = 1)

# train_set = train_set.drop(['Cabin'], axis = 1)
# test_set = test_set.drop(['Cabin'], axis = 1)


print(am)
print(ss)  
#############################아모레전처리####################################    


###########################################################################
am['일자'] = pd.to_datetime(am['일자'])

am['year'] = am['일자'].dt.strftime('%Y')
am['month'] = am['일자'].dt.strftime('%m')
am['day'] = am['일자'].dt.strftime('%d')


print(am)
am = am.drop(['일자'],axis=1)

print(am.shape)
print('=================')

cols = ['year','month','day']
for col in cols:
    le = LabelEncoder()
    am[col]=le.fit_transform(am[col])
##################################################################  
# size = 5 # x= 4개 y는 1개
# def split_x(dataset, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
#     aaa = []   #aaa 는 []라는 값이 없는 리스트임을 정의
#     for i in range(len(dataset)- size + 1): # 6이다 range(횟수)
#         subset = dataset[i : (i + size)]
#         #i는 처음 0에 개념 [0:0+size]
#         # 0~(0+size-1인수 까지 )노출 
#         aaa.append(subset) #append 마지막에 요소를 추가한다는 뜻
#     return np.array(aaa)    


# bbb = split_x(am, size)

# x1 = bbb[:,:-1]
# y1 = bbb[:,-1]

# print(x1.shape) # (1031, 4, 14)
# print(y1.shape) # (1031, 14)

# x1 = x1.reshape(1031, 4, 14)
# y1 = y1.reshape(1031, 14, 1)
# print(x1.shape) # (420547, 4, 14)
# print(y1.shape) # (96,1,1)
# # print(z.shape) # (6, 4)
###########################삼성전처리#########################################


ss['일자'] = pd.to_datetime(ss['일자'])

ss['year'] = ss['일자'].dt.strftime('%Y')
ss['month'] = ss['일자'].dt.strftime('%m')
ss['day'] = ss['일자'].dt.strftime('%d')


print(ss)
ss = ss.drop(['일자'],axis=1)

print(ss.shape)
print('=================')

cols = ['year','month','day']
for col in cols:
    le = LabelEncoder()
    ss[col]=le.fit_transform(ss[col])
    
#############################################################################
size = 6
def split_x(dataset2, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa2 = []   #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset2)- size + 1): # 6이다 range(횟수)
        subset2 = dataset2[i : (i + size)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa2.append(subset2) #append 마지막에 요소를 추가한다는 뜻
    return np.array(aaa2)    


aaa = split_x(am,size) #x1를 위한 데이터 drop 제외 모든 amo에 데이터
bbb = split_x(aa,size) #Y를 위한 시가만있는 데이터
x1 = aaa[:,:-3]
y = bbb[:,-3:]
ccc = split_x(ss,size) #x2를 위한 데이터 drop 제외 모든 Sam에 데이터
x2 = ccc[:,:-3]

print(x1,x1.shape) # ((1030, 3, 11)
print(x2,x2.shape) # ((1030, 3, 11)
print(y,y.shape) #(1030, 3)

# bbb2 = split_x(ss, size2)

# x2 = bbb2[:,:-1]
# y2 = bbb2[:,-1]

# print(x2.shape) # (1031, 4, 14)
# print(y2.shape) # (1031, 14)


# x2 = x2.reshape(1031, 4, 14)
# y2 = y2.reshape(1031, 14, 1)
# print(x2.shape) # (420547, 4, 14)
# print(y2.shape) # (96,1,1)
# # print(z.shape) # (6, 4)

# # am = am.reshape(3040, 21,1)
# # ss = ss.reshape(3180, 21,1)

# print(y.shape)
# print(am.shape)
# print(ss.shape)
# print(am.shape,ss.shape,y.shape)

#################################################################################
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y,
                                                    train_size=0.7, 
                                                    random_state=58525,shuffle=False
                                                    )

print(x1_train.shape,x1_test.shape)     # (721, 3, 11) (309, 3, 11)
print(x2_train.shape,x2_test.shape)     # (721, 3, 11) (309, 3, 11)
print(y_train.shape,y_test.shape)       # (721, 3) (309, 3)


# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit([x1_train,x2_train])
# # scaler.transform(x_test)
# x_test =scaler.transform([x1_test,x2_test])
# x_train = scaler.transform([x1_train,x2_train])

from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM,Dense,Dropout,Reshape,Conv1D
from tensorflow.python.keras.layers import Input
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
# scaler = StandardScaler()
# # scaler = MinMaxScaler()
# x1_train = x1_train.reshape(924,180)
# x1_test = x1_test.reshape(92,180)
# x2_train = x2_train.reshape(924,180)
# x2_test = x2_test.reshape(92,180)
# x1_train = scaler.fit_transform(x1_train)
# x2_train = scaler.fit_transform(x2_train)
# x1_test = scaler.transform(x1_test)
# x2_test = scaler.transform(x2_test)
# x1_train = x1_train.reshape(924,18,10)
# x1_test = x1_test.reshape(92,18,10)
# x2_train = x2_train.reshape(924,18,10)
# x2_test = x2_test.reshape(92,18,10)
# print(x1_train,x1_train.shape) #(924, 18, 10)
# print(x1_test,x1_test.shape) #(92, 18, 10)
# print(x2_train,x2_train.shape) # (924, 18, 10)
# print(x2_test,x2_test.shape) # (92, 18, 10)
# print(y_train.shape) #(924,)


#2. 모델구성

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input ,Dense

input1 = Input(shape=(3, 11))
dense1 = LSTM(128, activation= 'relu', name ='ys1')(input1)
dense2 = Dropout(0.2)(dense1)
dense3 = Dense(64, activation= 'relu', name ='ys2')(dense2)
dense4 = Dropout(0.2)(dense3)
dense5 = Dense(32, activation= 'relu', name ='ys3')(dense4)
dense6 = Dropout(0.2)(dense5)
dense7 = Dense(16, activation= 'relu', name ='ys4')(dense6)
dense8 = Dropout(0.2)(dense7)
output1 = Dense(10, activation= 'relu', name ='out_ys1')(dense8)

#2-2 
input2 = Input(shape=(3,11))
dense11 = LSTM(128, activation= 'relu', name ='ys11')(input2)
dense12 = Dropout(0.2)(dense11)
dense13 = Dense(64, activation= 'relu', name ='ys12')(dense12)
dense14 = Dropout(0.2)(dense13)
dense15 = Dense(32, activation= 'relu', name ='ys13')(dense14)
dense16 = Dropout(0.2)(dense15)
dense17 = Dense(16, activation= 'relu', name ='ys14')(dense16)
dense18 = Dropout(0.2)(dense17)
output2 = Dense(10, activation= 'relu', name ='out_ys12')(dense18)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2],name ='mg1')  
merge2 = Dropout(0.2)(merge1)     
merge3 = Dense(128, activation= 'relu',name ='mg2')(merge2)
merge4 = Dropout(0.2)(merge3)
merge5 = Dense(64, name ='mg3')(merge4)
merge6 = Dropout(0.2)(merge5)
merge7 = Dense(32, name ='mg4')(merge6)
merge8 = Dropout(0.2)(merge7)
merge9 = Dense(16, name ='mg5')(merge8)
merge10 = Dropout(0.2)(merge9)
merge11 = Dense(8, name ='mg6')(merge10)
merge12 = Dropout(0.2)(merge11)
last_output = Dense(1,activation='relu', name ='last')(merge12)

model =Model(inputs =[input1, input2], outputs= last_output)
model.summary()

# input1 = Input(shape=(4,11))
# dense1 = Dense(128, activation= 'relu', name ='ys1')(input1)
# dense2 = Dense(64, activation= 'relu', name ='ys2')(dense1)
# dense3 = Dense(32, activation= 'relu', name ='ys3')(dense2)
# dense4 = Dense(16, activation= 'relu', name ='ys4')(dense3)
# output1 = Dense(10, activation= 'relu', name ='out_ys1')(dense4)

# #2-2 
# input2 = Input(shape=(4,11))
# dense11 = Dense(128, activation= 'relu', name ='ys11')(input2)
# dense12 = Dense(64, activation= 'relu', name ='ys12')(dense11)
# dense13 = Dense(32, activation= 'relu', name ='ys13')(dense12)
# dense14 = Dense(16, activation= 'relu', name ='ys14')(dense13)
# output2 = Dense(10, activation= 'relu', name ='out_ys12')(dense14)

# from tensorflow.python.keras.layers import concatenate
# mergel = concatenate([output1, output2],name ='mg1')        # 단순하게 엮는것.      
# merge2 = Dense(128, activation= 'relu',name ='mg2')(mergel)
# merge3 = Dense(64, name ='mg3')(merge2)
# merge4 = Dense(32, name ='mg4')(merge3)
# merge5 = Dense(16, name ='mg5')(merge4)
# merge6 = Dense(1, name ='mg6')(merge5)
# last_output = Dense(1, name ='last')(merge6)

# model =Model(inputs =[input1, input2], outputs= last_output)


#####################################################################
print(x1_test.shape,x2_test.shape)

print(y_test.shape)

import datetime
date = datetime.datetime.now()
print(date)
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)


# #3. 컴파일,훈련


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    # < fit-callbacks에 있다.

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')           # 0707_1723
print(date)

filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

model.compile(loss='mae', optimizer='adam')
 
earlystopping =EarlyStopping(monitor='loss', patience=30, mode='min', 
              verbose=1, restore_best_weights = True)     
                      
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
                      save_best_only=True, 
                      filepath="".join([filepath,'k24_', date, '_', filename])
                    )
                                                                
model.compile(loss='mse', optimizer='Adam')
model.fit([x1_train,x2_train], y_train, 
          validation_split=0.3, 
          epochs=170,verbose=2
          ,batch_size=64
          ,callbacks=[earlystopping,mcp])    
        
model.save_weights("./_test/keras46_1_save_weights종가22.h5")
# model.load_weights("./_test/keras46_1_save_weights종가13.h5")

#4. 평가, 예측
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

loss = model.evaluate([x1_test, x2_test], y_test)       # (310, 4, 12) (310, 4, 12)   / (310,)
print('loss :', loss)

y_predict = model.predict([x1_test, x2_test])
print(y_predict)
print('종가 : ', y_predict[-1])
# print(y_predict.shape)    # (311, 1)

# def RMSE(a, b): 
#     return np.sqrt(mean_squared_error(a, b))
# rmse = RMSE(y_test, y_predict)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print('loss : ', loss)
# print("RMSE : ", rmse)
# print('r2스코어 : ', r2)



# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (6493, 1)
# submission_set = pd.read_csv(path + 'Submission.csv', # + 명령어는 문자를 앞문자와 더해줌
#                              index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(submission_set)
# submission_set['count'] = y_summit
# print(submission_set)
# submission_set.to_csv(path + 'submission.csv', index = True)


