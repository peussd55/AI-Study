### <<31>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=4325, stratify=y
)

# scikit-learn 분류모델은 입력데이터가 전부 1차원이어야한다.(y 원핫인코딩 X, but x 데이터 스케일러는 적용가능)
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
    
# LinearSVC : 0.9583333333333334
# LogisticRegression : 0.9694444444444444
# DecisionTreeClassifier : 0.8527777777777777
# RandomForestClassifier : 0.9805555555555555
