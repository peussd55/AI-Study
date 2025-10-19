### <<23>>

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

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
model = Sequential()
# model.add(SimpleRNN(units=512, input_shape=(3, 1)))
# model.add(SimpleRNN(units=512, input_length=3, input_dim=1))  # (timesteps, feature)
model.add(SimpleRNN(units=512, input_dim=1, input_length=3))    # 파라미터명을 명시하면 순서바뀌어도 된다.

model.add(Dense(256, activation='relu'))         # rnn은 출력 차원이 줄어들기때문에(출력이 시계열이 꼭 아니다) cnn과 달리 flatten필요없이 Dense와 바로 연결할 수있다.
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000)

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1)    # (3,) -> (1,3,1)
y_pred = model.predict(x_pred)

print('[8,9,10]의 결과 : ', y_pred)
# 시계열모델은 데이터가 적으면 성능이 안나온다. 아래 결과는 사실상 Dense모델로 뽑아낸결과이다.
# RNN : [8,9,10]의 결과 :  [[11.00545]]
# LSTM : [8,9,10]의 결과 :  [[10.967978]]
# GRU : [8,9,10]의 결과 :  [[10.9556]]