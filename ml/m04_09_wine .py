### <<31>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True)) 

# scikit-learn 분류모델은 y 인코딩 하지 않는다. 내부적으로 다중 클래스 처리를 한다.
# # y 원핫인코딩
# y = y.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)   # nparray로 반환시키는 객체
# y = ohe.fit_transform(y)
# print(y.shape)  #(178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 517,
    stratify=y,
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
    
# LinearSVC : 0.9444444444444444
# LogisticRegression : 0.9444444444444444
# DecisionTreeClassifier : 1.0
# RandomForestClassifier : 1.0
