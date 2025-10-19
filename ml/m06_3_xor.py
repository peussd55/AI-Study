### <<31>>

"""
# 인공지능 1차겨울 : 선형회귀로는 XOR을 못 푸는 문제
# 선형회귀식 : y = w1x1 + w2x2 + ... + b
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import  LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델 (선형회귀)
model = Perceptron()    # 단층 퍼셉트론
# model = LinearSVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
results = model.score(x_data, y_data)
print('model.score :', results)

y_predict = model.predict(x_data)
acc = accuracy_score(y_data, y_predict)
print('accuracy score :', acc)

# model.score : 0.5
# accuracy score : 0.5

# model.score : 0.5
# accuracy score : 0.5

# => *** 선형회귀로는 xor문제를 못 풂 ***