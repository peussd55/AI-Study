### <<31>>

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
print(pd.value_counts(y))

# ML 분류모델은 y 원핫인코딩 하지 않는다.(input으로 1차원데이터만 받는다.)
# y = y.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# print(y.shape) # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state= 999,
    stratify=y,
)


# scikit-learn 분류모델은 입력데이터가 전부 1차원이어야한다.(y 원핫인코딩 X, but x 데이터 스케일러 적용가능)
# 2. 모델구성 (전부 분류)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_list = [LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

for index, value in enumerate(model_list):
    if value.__name__ == 'LogisticRegression':
        model = value(solver='liblinear')   # 경고로그안뜨게하려고 적용하는거임
    else:
        model = value()
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(f"{value.__name__} :", results)
    
# LinearSVC : 0.5744602118706057
# LogisticRegression : 0.7104291627582765
# DecisionTreeClassifier : 0.9406727881379999
# RandomForestClassifier : 0.9564641188265364
# 분류모델(LogisticRegression, DecisionTreeClassifier, RandomForestClassifier)이 회귀모델(LinearSVC)보다 성능이 확연히 좋다.