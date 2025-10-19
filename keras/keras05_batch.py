### <<03>>

### batch : 훈련할때 데이터를 쪼개는 단위 ###

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# batch_size : 학습데이터를 n분의 1만큼 쪼개서 훈련을 1회진행(행의 사이즈). 일반적으로 배치 사이즈가 작아질수록 성능이 좋아지고, 데이터가 쪼개져서 메모리에 올라가기때문에 자원효율적이다. 대신 그만큼 훈련횟수가 늘어난다.
# batch_size=1 : 1 -> 1 / 2 -> 2 / 3 -> 3 / ....
# batch_size=2 : [1,2] -> [1,2] / [3,4] -> [3,4] / [5,6] -> [5,6]
# batch_size=4 : [1,2,3,4] -> [1,2,3,4] / [5,6] -> [5,6] 
# 실제훈련 횟수 : (epochs) * (bath_size가 쪼개는 단위 갯수)
# fit에서 batch_size를 명시하지 않으면 리스트 크기만큼이 batch size가 된다. 전체 리스트 덩어리를 메모리에 올려서 한번만 학습. 이 경우 실제 훈련 횟수는 epcohs 만큼이다.

# 2. 모델구성 : 히든~출력레이어의 인풋은 어차피 상위레이어의 아웃풋이므로 문법적으로 생략가능
model = Sequential() 

## 설계 1 : 배치사이즈가 작아질수록 loss도 작아진다. (일반적인 케이스)
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# ## 설계 2 : 배치사이즈가 작아질수록 오히려 loss가 커진다. (과적합문제?)
# model.add(Dense(10, input_dim=1))
# model.add(Dense(50))
# model.add(Dense(100))
# model.add(Dense(25))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))

# 3. 컴파일, 훈련
epochs = 100
batch_size = 1
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=batch_size)

print("##############################################################################")
# 4. 평가, 예측
loss = model.evaluate(x, y)     # 훈련이끝난 모델을 기본값 배치사이즈(32size)로 loss를 한번 계산한다. 훈련(가중치 재조정)을 하는 것은 아니다. 03.손실함수.txt 참고
print('epochs :', epochs)
print('loss :', loss)
# results = model.predict([7])
# print('7의 예측값 :', results)

