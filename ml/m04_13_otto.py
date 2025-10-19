### <<31>>

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. 데이터
path = './_data/kaggle/otto/'

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# # target컬럼 레이블 인코딩(원핫 인코딩 사전작업)###
# # 정수형을 직접 원핫인코딩할경우 keras, pandas, sklearn 방식 모두 가능하지만 문자형태로 되어있을 경우에는 pandas방식만 문자열에서 직접 원핫인코딩이 가능하다.
# le = LabelEncoder() # 인스턴스화
# train_csv['target'] = le.fit_transform(train_csv['target'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용

# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
print('x type:',type(x))
y = train_csv['target']
print('y type:',type(y))

# # onehotencoding
# y = pd.get_dummies(y)
# print("Pandas get_dummies:")
# print(y.head())
# print(y.head(-5))    
# print(y.shape)      # (61878, 9)
# print(type(y))      # <class 'pandas.core.frame.DataFrame'>


# 데이터분할
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    random_state=8282,
    shuffle=True,
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
    
# LinearSVC : 0.7484647705235941
# LogisticRegression : 0.7543632837750485
# DecisionTreeClassifier : 0.7148513251454428
# RandomForestClassifier : 0.8102779573367809

