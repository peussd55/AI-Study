### <<26>>

# 52-1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Flatten

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([
                [1,2,3],    
                [2,3,4],
                [3,4,5],
                [4,5,6],
                [5,6,7],
                [6,7,8],
                [7,8,9],
            ])  # 7x3. timesteps은 3
y = np.array([4,5,6,7,8,9,10])
# dataset만 주어지고 y는 내가 정해야한다.(잘라야한다.)
print(x.shape, y.shape) # (7, 3) (7,)

# x를 3차원으로 변환
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (7, 3, 1)     // (batch_size, timesteps, feature)
# x = np.array([
#                 [[1],[2],[3]],    
#                  ....
#                 [[2],[3],[4]],
#                 [[7],[8],[9]],
#             ])
# feature : 훈련할때 timesteps 하나당 짤리는 횟수. feature가 1이면 timesteps하나가 y 하나에 훈련된다.

# 2. 모델구성
# Conv1D : cnn의 시계열 처리레이어
# Dense와 Conv1D의 차이 : Conv1D는 Dense에서 kernel_size만큼 가중치가 추가된거와 같다고 보면된다.
# Conv1D : 가중치가 행렬형태이고 행렬연산, stride에 따라 중첩연산됨
# Dense : 가중치가 선형연산
# Cov1D와 RNN의 차이 : RNN은 과거데이터를 추출 / Conv1D는 특성추출 -> Conv1D는 망각현상X
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=2, padding='same', input_shape=(3,1))) # input : (N, 3, 10) / output : (N, 3, 10)
model.add(Conv1D(9, 2)) # filter=9, kernel=2 // output : N, 2, 9
model.add(Flatten())    # output : N, 18
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1)    # (3,) -> (1,3,1)
y_pred = model.predict(x_pred)

print('[8,9,10]의 결과 : ', y_pred)
# [8,9,10]의 결과 :  [[11.009423]]
