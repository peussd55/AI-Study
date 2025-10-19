### <<04>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

# 1. 데이터
# np.array의 용도 : 행렬,텐서연산이 가능한 (중첩가능한)리스트로반환. [] 감싸서 반환한다.
x = np.array(range(10))     # range(0, 10) : (0,1,2,3,4,5,6,7,8,9)
print(x)        # [0,1,2,3,4,5,6,7,8,9]
print(x.shape)  # (10,) : 벡터, 리스트

x = np.array(range(1, 10))
print(x)        # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11))
print(x)        # [ 1  2  3  4  5  6  7  8  9 10]

x = np.array([range(10), range(21,31), range(201, 211)])
# x = np.array(range(10), range(21,31), range(201, 211)) 가 오류가 나는 이유 : 
# array는 ()내부에 인자를 2개까지만 받고 그 인자도 데이터형식에 맞아야한다. 첫번째는 덩어리데이터, 두번째는 그 데이터타입을 명시한 것이 와야하는데 정의되지 않은 세번째인자가 명시되었기때문이다.
print(x)
print(x.shape)  # (3, 10)

x = x.T         # == transpose(x) == 전치행렬
print(x)
print(x.shape)  # (10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습]
# [19, 31, 211] 예측

# 2. 모델구성 
# 레이어 구성할때 노드의 수는 순차적으로 증가하다 순차적으로 감소하는식으로 짜는게 좋다. (다이아몬드, 항아리모양)
# 중간이 움푹패진 식으로 구성하면 데이터 소실이 일어날 수 있다.
# 레이어 노드 수는 통상적으로 2의배수로 구성한다.
model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
epochs = 100
batch_size = 1
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=batch_size)

# 4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[10, 31, 211]])
print('loss :', loss)
print('[10, 31, 211]의 예측값:', results)
# loss : 2.3570537166506256e-07
# [10, 31, 211]의 예측값: [[10.998942]]