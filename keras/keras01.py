### <<01>>

import tensorflow as tf
print(tf.__version__)   # 2.9.3
import numpy as np
print(np.__version__)   # 1.21.1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

# 1. 데이터(전처리)
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델구성
model = Sequential()                            # Sequential(레이어를 순차적으로 쌓음)이라는 모델을 만듦. 그 이름은 model이라고 정의
model.add(Dense(1, input_dim=1))                # Dense(완전연결레이어 : 인풋 레이어 노드 하나 당 전 레이어의 노드를 모두 연결)함수로 파라미터에서 아웃풋 1개, 인풋은 1개로 정의. Dense가 모델을 y = x로 정의.

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')       # 컴파일 : 컴퓨터가 알아먹는 형태로 변환. 손실함수는 mse, optimizer(경사하강법, 역전파)는 adam 이라는 것을 사용
model.fit(x,y, epochs=10000)                      # 데이터 x,y를 넣고 10000번 반복훈련. "x는 y다" 라는 훈련을 10000번시킴. 가중치가 갱신/생성됨.

# 4. 평가, 예측
result = model.predict(np.array([5]))   # y = wx + b. 여기서 쓰이는 가중치는 최적의 w가 아니고 epochs횟수(10000)만큼 훈련돌린 모델의 w이다.
print('5의 예측값 : ', result)