### <<31>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=8282, shuffle=True
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
    
# LinearSVC : 0.5239494652649438
# LogisticRegression : 0.7834095797861059
# DecisionTreeClassifier : 0.7984063986427121
# RandomForestClassifier : 0.8573635895416124
