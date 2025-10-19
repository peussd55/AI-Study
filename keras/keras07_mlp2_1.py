### <<03>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])    # (3, 10) : 잘못된 데이터

y = np.array([1,2,3,4,5,6,7,8,9,10])   
x = np.transpose(x)                     # (3, 10) -> (10, 3) / y가 10개라서 데이터갯수(행)도 10개여야한다.

print(x.shape)
print(y.shape)

# 2. 모델구성
model = Sequential()
## 행 무시, 열 우선 // 열=컬럼=차원=feature // 행 = 데이터 갯수
model.add(Dense(10, input_dim=3))       # input_dim은 x.shape를 찍어서 확인한 열(컬럼)과 같은 수가 들어가야한다.
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[11,2.0,-1],[12,3.0,-2]])       # (2, 3) : x의 컬럼(3)과 동일한 컬럼(3) 
print('loss :', loss)
print('[11,2.0,-1], [12,3.0,-2]의 예측값 :', results)     # epoch가 작은데도 예측값이 잘 나온다. 즉, 컬럼(또는 행(데이터))이 많을 수록 성능도 좋다.