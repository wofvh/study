from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


datasets =load_breast_cancer()

# x = datasets.data
# y = datasets.target

df = pd.DataFrame(datasets.data,columns=[datasets.feature_names])
print(df)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# datasets.target = le.fit_transform(datasets.target)

kmeans =KMeans(n_clusters=2, random_state=50)
kmeans.fit(df)


results = kmeans.labels_
print(results)
print(datasets.target)

df['cluster'] = kmeans.labels_
df['target'] = datasets.target

#4. 평가


score = accuracy_score(df['cluster'],df['target'])
print('결과',score)


# [0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 2 2 0 0 2 0 0 0 0 0 0 2 2
#  0 0 2 2 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 2 1 1 2 1 1 2 2 2 1 1 0
#  2 1 1 1 2 1 1 2 2 1 1 1 1 1 2 2 1 1 1 1 1 2 2 1 2 1 2 1 1 1 2 1 1 1 1 2 1
#  1 2 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 2 1 1 2 2 2 2 1 1 1 2 2 1 1 2 2 1 2
#  2 1 1 1 1 2 2 2 1 2 2 2 1 2 1 2 2 1 2 2 2 2 1 1 2 2 2 2 2 1]
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
# 결과 0.702247191011236
