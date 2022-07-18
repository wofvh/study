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
am = pd.read_csv( path + '아모레.csv', thousands=',')
ss = pd.read_csv( path + '삼성.csv', thousands=',')


am.at[1035:,'시가'] = 0
print(am) #2018/05/04

ss.at[1035:,'시가'] = 0
# print(ss) #2018/05/04

ss = ss[ss['시가'] > 100] #[1035 rows x 17 columns]
print(ss.shape)
print(ss)
am = am[am['시가'] > 100] #[1035 rows x 17 columns]
print(am.shape,ss.shape)
print(am) #2018/05/04
print(am,ss)      # (3040, 17) (3180, 17)  


y = np.array(am['시가'])   
print(y.shape)   # [1035 rows x 17 columns]  (1035,)


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

am = pd.DataFrame(data=am)
am = am.drop(columns=['전일비','증감량','외국계','프로그램','외인비'])
ss = ss.drop(columns=['전일비','증감량','외국계','프로그램','외인비'])
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
size = 5 # x= 4개 y는 1개
def split_x(dataset, size): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa = []   #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset)- size + 1): # 6이다 range(횟수)
        subset = dataset[i : (i + size)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa.append(subset) #append 마지막에 요소를 추가한다는 뜻
    return np.array(aaa)    


bbb = split_x(am, size)

x1 = bbb[:,:-1]
y1 = bbb[:,-1]

print(x1.shape) # (1031, 4, 14)
print(y1.shape) # (1031, 14)

x1 = x1.reshape(1031, 4, 14)
y1 = y1.reshape(1031, 14, 1)
print(x1.shape) # (420547, 4, 14)
print(y1.shape) # (96,1,1)
# print(z.shape) # (6, 4)
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
size2 = 5 
def split_x(dataset2, size2): # def라는 예약어로 split_x라는 변수명을 아래에 종속된 기능들을 수행할 수 있도록 정의한다.
    aaa2 = []   #aaa 는 []라는 값이 없는 리스트임을 정의
    for i in range(len(dataset2)- size2 + 1): # 6이다 range(횟수)
        subset2 = dataset2[i : (i + size2)]
        #i는 처음 0에 개념 [0:0+size]
        # 0~(0+size-1인수 까지 )노출 
        aaa2.append(subset2) #append 마지막에 요소를 추가한다는 뜻
    return np.array(aaa2)    


bbb2 = split_x(ss, size2)

x2 = bbb2[:,:-1]
y2 = bbb2[:,-1]

print(x2.shape) # (1031, 4, 14)
print(y2.shape) # (1031, 14)


x2 = x2.reshape(1031, 4, 14)
y2 = y2.reshape(1031, 14, 1)
print(x2.shape) # (420547, 4, 14)
print(y2.shape) # (96,1,1)
# print(z.shape) # (6, 4)

# am = am.reshape(3040, 21,1)
# ss = ss.reshape(3180, 21,1)

print(y.shape)
print(am.shape)
print(ss.shape)
print(am.shape,ss.shape,y.shape)

#################################################################################
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(am,ss,y,
                                                    train_size=0.7, 
                                                    random_state=66,shuffle=False
                                                    )

print(x1_train.shape,x1_test.shape)     # (724, 14) (311, 14)
print(x2_train.shape,x2_test.shape)     # (724, 14) (311, 14)
print(y_train.shape,y_test.shape)       # (724,) (311,)


# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit([x1_train,x2_train])
# # scaler.transform(x_test)
# x_test =scaler.transform([x1_test,x2_test])
# x_train = scaler.transform([x1_train,x2_train])

#2. 모델구성

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input ,Dense

input1 = Input(shape=(14,))
dense1 = Dense(128, activation= 'relu', name ='ys1')(input1)
dense2 = Dense(64, activation= 'relu', name ='ys2')(dense1)
dense3 = Dense(32, activation= 'relu', name ='ys3')(dense2)
dense4 = Dense(16, activation= 'relu', name ='ys4')(dense3)
output1 = Dense(10, activation= 'relu', name ='out_ys1')(dense4)

#2-2 
input2 = Input(shape=(14,))
dense11 = Dense(128, activation= 'relu', name ='ys11')(input2)
dense12 = Dense(64, activation= 'relu', name ='ys12')(dense11)
dense13 = Dense(32, activation= 'relu', name ='ys13')(dense12)
dense14 = Dense(16, activation= 'relu', name ='ys14')(dense13)
output2 = Dense(10, activation= 'relu', name ='out_ys12')(dense14)

from tensorflow.python.keras.layers import concatenate
mergel = concatenate([output1, output2],name ='mg1')        # 단순하게 엮는것.      
merge2 = Dense(128, activation= 'relu',name ='mg2')(mergel)
merge3 = Dense(64, name ='mg3')(merge2)
merge4 = Dense(32, name ='mg4')(merge3)
merge5 = Dense(16, name ='mg5')(merge4)
merge6 = Dense(8, name ='mg6')(merge5)
last_output = Dense(1, name ='last')(merge6)

model =Model(inputs =[input1, input2], outputs= last_output)
model.summary()

#3. 컴파일,훈련
model.compile(loss='mae', optimizer='adam')
 
earlystopping =EarlyStopping(monitor='loss', patience=100, mode='auto', 
              verbose=1, restore_best_weights = True)     
                                                                
hist = model.fit([x1_test, x2_test], y_train, epochs =5000, batch_size = 32, 
                 verbose=1, 
                 validation_split = 0.3,
                 callbacks = [earlystopping])     
        

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss :', loss)

y_predict = model.predict([x1_test, x2_test])
print(y_predict)

print(y_predict.shape)    # (311, 1)

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

