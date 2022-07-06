from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 대문자 class  암시가능.
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target
 
 
print(np.min(x))
print(np.max(x))
x = (x - np.min(x) / (np.max(x)) - np.min(x))  # 0~1 사이가 된다. 
print(x[:10])
