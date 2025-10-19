### <<08>>

# 14 카피

import numpy as np 
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense 
import time     # 시간에 대한 모듈 import

# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10]).T  # 벡터는 transpose를 해도 변화가 없다.
# y = np.array([1,2,3,4,5,6,7,8,9,10]).T  # 벡터는 trnaspose를 해도 변화가 없다.

# print(x)
# print(y)

# print(x.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산
# print(y.shape)  #(10,)  엄밀히 말하면 (10,1)과 다르지만 (10, 1)처럼 받아들이고 연산

# #exit()
# 훈련데이터(70%)
x_train = np.array(range(100))
y_train = np.array(range(100))

# 평가데이터(30%)
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))    # input 차원이 1(x컬럼 1)이면서 출력 1개(y컬럼 1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()  # 현재 시간을 반환, 시작시간.(1970년 1월 1일 00:00 이후로 지난 시간을 초단위로 반환)
print(start_time)         # 1747968690.0155628
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          verbose=1,
          )
# 훈련과정을 로그로 보여주는 건 실제 훈련속도와 로그를 프린트해서 보여주는 속도의 괴리감으로 인해 전체훈련속도에 영향을 준다. 따라서 옵션으로 켜고 끌수있다.
# verbose 옵션으로 로그를 끄면 훈련 한번당 로그프린트를 할 필요가 없기때문에 학습속도를 더빨리 할 수있다.
# 0 : 침묵
# 1 : 디폴트
# 2 : 프로그레스바 삭제
# 3 : 에포만 나옴
end_time = time.time()
print('걸린시간 :', end_time - start_time, '초')

# epochs : 1000 // batch : 1
# 0 : 35.56377291679382 초
# 1 : 45.62639093399048 초
# 2 : 36.488988399505615 초
# 3 : 37.84654355049133 초

# epochs : 1000 // batch : 32
# 0 : 2.154406785964966 초
# 1 : 2.8764188289642334 초
# 2 : 2.3311173915863037 초
# 3 : 2.2004899978637695 초

# epochs : 1000 // batch : 128
# 0 : 0.9706449508666992 초
# 1 : 1.4854505062103271 초
# 2 : 1.2179453372955322 초
# 3 : 1.1320726871490479 초
