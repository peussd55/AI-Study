### <<05>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

# 1. 데이터
x = np.array([range(10)])   #(1,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [10,9,8,7,6,5,4,3,2,1],
             [9,8,7,6,5,4,3,2,1,0]])

print(x)
print(y)
print(y.shape)  # (3,10)
x = x.T         # (10,1)
y = y.T
print(y.shape)  # (10,3)
print(x)
print(y)

# 2. 모델구성
model = Sequential() 
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)     # 평가데이터가 훈련데이터와 동일하다는 문제점이 있다.(답지를 알고있는 상태에서 시험 보는 것과 비슷)
results = model.predict([[10]]) # (1,1)
print('loss :', loss)
print('[10]의 예측값 :', results)

# loss : 5.159724101970464e-13
# [10] : [[ 1.1000000e+01  1.4156103e-06 -9.9999976e-01]]    [11, 0, -1]