### <<32>>

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
print(x)
print(y)

df =pd.DataFrame(x, columns=datasets.feature_names) # nparray -> pd.dataframe
print(df)

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=True)   # shuffle=True : validation이 3분의 1비율을 유지하면서 겹쳐지지않게 3등분하여 선택됨

for index, (train_index, val_index) in enumerate(kfold.split(df)):
    print(f"===============[{index+1}]===============")     
    print(train_index, '\n', val_index)