### <<04>>

import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])       # (3,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10], [10,9,8,7,6,5,4,3,2,1]])  # (2,10)
x = x.T
y = y.T
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print(x)
print(y)
print(x.shape)  # (3, 10) -> (10, 3) : input의 컬럼은 3개
print(y.shape)  # (2, 10) -> (10, 2) : output의 컬럼은 2개 -> 읽는법(Nan by two) : 행의 갯수는 중요하지 않아서 Nan이라고 읽고 ,는 by라고 읽는다.
# x의 행의 수와 y의 행의수만 일치하면된다.

# [실습]
# loss와 [[10, 31, 211]]을 예측하시오

# 2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=3))    # input_dim은  x의 컬럼수와 반드시 일치해야한다.
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))     # y의 컬럼이 2이기때문에 출력레이어의 Dense는 2로 지정하여야한다.
# y 컬럼이 1일 경우에는 출력레이어 Dense가 1이 아니어도 브로드캐스팅되어(y를 차원 확장 : [1] -> [1,1]) 같은 결과 값을 여러번 출력하지만 
# y컬럼이 1이 아니면 브로드캐스팅이 되지 않아 출력 Dense가 일치하지 않으면 오류가 발생한다.

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=1)   # epochs가 100밖에 안되면 훈련돌릴때마다 loss가 들쭉날쭉하다.

# 4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict([[10, 31, 211], [11,32,212]]) 
print('loss :', loss)
print('[10, 31, 211], [11,32,212]의 예측값 :', results)

# [10, 31, 211], [11,32,212]s의 예측값 : [[ 1.0999955e+01  8.1814826e-05]
#  [ 1.1999928e+01 -9.9991244e-01]]