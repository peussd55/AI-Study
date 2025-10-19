### <<08>>

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 

#[실습]
#r2_score > 0.62

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    #test_size=0.3,
    random_state=100
)

# 2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1)) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=1, validation_split=0.1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)

"""
r2 스코어 : 0.5237661957529425 -> validation을 0.1로 했을때 validation없을때(0.51)보다 0.01향상.
"""