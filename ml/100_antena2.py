import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 
import random

# def seed_everything(seed):
#       random.seed(seed)
#   os.environ['PYTHONHASHEED'] = str(seed)
#   np.random.seed(seed)
# seed_everything(42)

train = pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/test.csv")
submit= pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/sample_submission.csv")

print(train.head(5))

x_feature_info = pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/x_feature_info.csv")
print(x_feature_info)

y_feature_info = pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/y_feature_info.csv")
print(y_feature_info)

y_feature_spec_info = pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/y_feature_spec_info.csv")
print(y_feature_spec_info)

train_x = train.filter(regex='X')
train_y = train.filter(regex = "Y")

#test_x
test = pd.read_csv("https://raw.githubusercontent.com/annsyj94/Data_Analytics_Portfolio/main/lg_aimers/test.csv").drop(columns = ['ID'])
print(test.head(5))

pcb = train_x[['X_01','X_02','X_05','X_06']]

# PCB 체결 시 단계별 누름량
fig, axes = plt.subplots(2,2, figsize = (15,10))

sns.histplot(data = pcb, x = "X_01", kde = True, ax = axes[0,0]).set(title = "PCB 체결 시 단계별 누름량(1)")
sns.histplot(data = pcb, x = "X_02", kde = True, ax = axes[0,1]).set(title = "PCB 체결 시 단계별 누름량(2)")
sns.histplot(data = pcb, x = "X_05", kde = True, ax = axes[1,0]).set(title = "PCB 체결 시 단계별 누름량(3)")
sns.histplot(data = pcb, x = "X_06", kde = True, ax = axes[1,1]).set(title = "PCB 체결 시 단계별 누름량(4)")

plt.show()