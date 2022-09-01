import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer,load_iris
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA

data = load_iris()
df = pd.DataFrame(data.data,columns =data.feature_names)
df['target'] = pd.Series(data.target,dtype='category').cat.rename_categories(data.target_names)

df.head(3)

colors =['navy','black','red']
for xy,target in zip(data.data,data.target):
    plt.scatter(x=xy[0],y=xy[1],color = colors[target])
    plt.show()
    