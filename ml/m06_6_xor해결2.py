### <<31>>

"""
# 은닉층을 추가한 다층퍼셉트론모델로 XOR문제 해결 (비선형성을 부여하는 활성화함수가 들어가야한다)
# 종이를 접으면 층이 나뉘듯이 층을 나누면 1만 또는 0만 추출할 수 있음.
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import  LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = SVC()
model = DecisionTreeClassifier()

# 3. 컴파일, 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
results = model.score(x_data, y_data)
print('model.score :', results)

y_predict = model.predict(x_data)
acc = accuracy_score(y_data, np.round(y_predict))
print('accuracy score :', acc)

# model.score : 1.0
# accuracy score : 1.0

# model.score : 1.0
# accuracy score : 1.0
