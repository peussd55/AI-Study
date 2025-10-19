### <<03>>

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# epoch는 100으로 고정
# loss는 0.32 미만으로 만들 것.
# 모델은 일직선으로 된 일차함수를 그리는데 데이터분포 그래프에서 꺾이는 부분이 있으면 일정 오차가 반드시 발생할 수 밖에없다.

# 2. 모델구성 : 히든~출력레이어의 인풋은 어차피 상위레이어의 아웃풋이므로 문법적으로 생략가능
model = Sequential() 
model.add(Dense(10, input_dim=1))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
epochs=100
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

print("##############################################################################")
# 4. 평가, 예측
loss = model.evaluate(x, y)
print('epochs :', epochs)
print('loss :', loss)
# results = model.predict([7])
# print('7의 예측값 :', results)

# epochs : 100
# loss : 0.32384786009788513