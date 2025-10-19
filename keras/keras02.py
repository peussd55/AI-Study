### <<01>>

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 

# 1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,4,5,6])

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)     # 100번째 훈련한 가중치가 들어가 있는것이지 100번 중에 제일 좋은 가중치가 들어가있는것이 아니다.

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
# 4. 평가, 예측
loss = model.evaluate(x, y)       # 훈련 완료된 모델의 loss를 검증차원에서 한번 계산. 훈련을 더 시키는 것은 아니다.(가중치 재조정 하는 것 아님)
print('로스 : ', loss)
result = model.predict([1,2,3,4,5,6,7])
print('예측값 : ', result)