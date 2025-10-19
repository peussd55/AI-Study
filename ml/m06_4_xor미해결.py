### <<31>>

"""
# 단층 퍼셉트론으로는 XOR 해결불가
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import  LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델 (선형회귀)
# model = Perceptron()    # 단층 퍼셉트론
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) 
# -> 단층 퍼셉트론으로는 xor해결불가 (뭔짓을 해도 acc 0.75를 못넘는다.)

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=100)

# 4. 평가, 예측
results = model.evaluate(x_data, y_data)
print('model.evaluate :', results)

y_predict = model.predict(x_data)
acc = accuracy_score(y_data, np.round(y_predict))
print('accuracy score :', acc)
# accuracy score : 0.5
