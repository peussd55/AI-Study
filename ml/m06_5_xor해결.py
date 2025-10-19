### <<31>>

"""
# 은닉층을 추가한 다층퍼셉트론모델로 XOR문제 해결 (비선형성을 부여하는 활성화함수가 들어가야한다)
# 종이를 접으면 층이 나뉘듯이 층을 나누면 1만 또는 0만 추출할 수 있음.
"""

from sklearn.linear_model import Perceptron
from sklearn.svm import  LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 랜덤고정(cpu연산에서는 고정되더라도 gpu 가중치연산에서 틀어질수있다)
import random
import numpy as np
import tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델 (선형회귀)
# model = Perceptron()    # 단층 퍼셉트론
# model = LinearSVC()
model = Sequential()
model.add(Dense(10, input_dim=2, activation='sigmoid')) 
# 아무리 레이어를 많이 쌓아도 은닉층에 활성화함수가 들어가지 않으면 선형 연산만 수행하기때문에 비선형적으로 구분해야하는 XOR문제를 해결할 수없다.
model.add(Dense(1, activation='sigmoid')) 

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=1000)

# 4. 평가, 예측
results = model.evaluate(x_data, y_data)
print('model.evaluate :', results)

y_predict = model.predict(x_data)
acc = accuracy_score(y_data, np.round(y_predict))
print('accuracy score :', acc)
# accuracy score : 1.0