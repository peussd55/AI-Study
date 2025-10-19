### <<03>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

# 1. 데이터
x = np.array([[1,2,3,4,5],
             [6,7,8,9,10]]) # (2, 5) : 행(데이터 갯수)이 2개인데 이대로 진행하면 y는 5개의 요소(데이터갯수)를 가지고있기때문에 1대1 매칭이 되지않아 모델 학습시 오류가 난다.

y = np.array([1,2,3,4,5])   # (5,)의 데이터 갯수는 5이다. 벡터이기때문에 행렬과 표현구조가 다를뿐이다.
# x = np.array([1,6],[2,7],[3,8],[4,9],[5,10])
x = np.transpose(x)         # (5, 2) : transpose는 (2,5)을 (5,2)로 변환. -> 데이터갯수 5개

print(x.shape)
print(y.shape)

# 2. 모델구성
model = Sequential()
## 행 무시, 열 우선 // 열=컬럼=차원 // 행 = 데이터 갯수
model.add(Dense(10, input_dim=2))   # input_dim은 x.shape를 찍어서 확인한 열(컬럼)과 같은 갯수가 들어가야한다.
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
results = model.predict([[6,11]])      # (1, 2) : predict의 열의 갯수는 input_dim 개숫와 맞으면 된다. 즉, 모델의 구조(컬럼)과 예측값의 구조(컬럼)이 맞기만 오류가 안난다.
print('loss :', loss)
print('[6,11]의 예측값 :', results)     # epoch가 작은데도 예측값이 잘 나온다. 즉, 컬럼이 많을 수록 성능도 좋다.

