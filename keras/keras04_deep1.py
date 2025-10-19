### <<02>>

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델구성 (하이퍼파라미터 튜닝 : 레이어를 늘리거나 노드를 늘릴 수있다. 인접한 히든레이어간 입출력 노드갯수가 맞아야한다. 오류가 나진 않는다.)
model = Sequential()
model.add(Dense(3, input_dim=1))    # 첫번째 히든레이어(input 1개, output 3개) : 첫번째 히든레이어의 입력은 입력레이어이고
model.add(Dense(4, input_dim=3))    # 두번째 히든레이어(input 3개, output 4개) : 두번째 히든레이어의 입력은 첫번째 히든레이어이고
model.add(Dense(2, input_dim=4))    # 세번째 히든레이어(input 4개, output 2개) : 세번째 히든레이어의 입력은 두번째 히든레이어이고

model.add(Dense(30, input_dim=2))     # 추가된 히든레이어
model.add(Dense(40, input_dim=30))    # 추가된 히든레이어
model.add(Dense(50, input_dim=40))    # 추가된 히든레이어
model.add(Dense(70, input_dim=50))    # 추가된 히든레이어
model.add(Dense(2, input_dim=70))     # 추가된 히든레이어

model.add(Dense(1, input_dim=2))    # 출력레이어(input 2개, output 1개) : 출력레이어의 입력은 세번째 히든레이어이다.

# 3. 컴파일, 훈련
epochs = 300  # (하이퍼파라미터 튜닝 : epochs 수를 늘리거나 줄일 수 있다.)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

print("##############################################################################")
# 4. 평가, 예측
loss = model.evaluate(x, y)     # loss를 한번 계산한다.
print('epochs :', epochs)
print('loss :', loss)
#results = model.predict([1,2,3,4,5,6,7])
results = model.predict([6])
print('6의 예측값 :', results)

# epochs : 300
# loss : 2.452692626775388e-08
# 6의 예측값 : [[5.999759]]