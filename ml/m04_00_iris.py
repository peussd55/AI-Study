### <<31>>

import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x, y = load_iris(return_X_y=True)
# print(x)
# print(y)
print(x.shape, y.shape) # (150, 4) (150,)

# 2. 모델구성 (전부 분류)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression     # 얘는 이름에 회귀(Regression)가 들어가지만 이진분류 모델이다.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

# model = LinearSVC(C=0.3)
# model = LogisticRegression()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

# 3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy',   # 정수 y에 대하여 자동 원핫인코딩해주는 다중분류 손실함수
#               optimizer='adam',
#               metrics=['acc'],
#               )
# model.fit(x, y, epochs=100)

model.fit(x, y)

# 4. 평가, 예측
# results = model.evaluate(x, y)
results = model.score(x, y)
# sklearn 머신러닝 모델의 디폴트 평가지표 : 회귀(r2_score), 분류(accracy)

print(results)

# DecisionTreeClassifire : 1.0