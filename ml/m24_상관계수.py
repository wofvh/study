import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target

df = pd.DataFrame(x, 
                  columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
#pd.DataFrame(열에 이름을 지정해준다)
# print(df)

df['Target(Y)']= y
# df에 Target(Y)를 생성후 y 값을 넣는다. 

print(df)
print("=============================상관계수 히트 맵==============")
print(df.corr())                    # 상관관계를 확인.  
# [150 rows x 5 columns]
#               sepal length  sepal width  petal length  petal width  Target(Y)
# sepal length      1.000000    -0.117570      0.871754     0.817941   0.782561
# sepal width      -0.117570     1.000000     -0.428440    -0.366126  -0.426658
# petal length      0.871754    -0.428440      1.000000     0.962865   0.949035
# petal width       0.817941    -0.366126      0.962865     1.000000   0.956547
# Target(Y)         0.782561    -0.426658      0.949035     0.956547   1.000000

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),square=True, annot=True, cbar=True) 
# cbar 옆에 생기는 바형태 모형.
# 상관관계를 보여주는 맵.  

plt.show()
