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
# model.add(Dense(units=10, input_dim=3))
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
# model.add(LSTM(units=10, input_shape=(3, 1)))
model.add(GRU(units=10, input_shape=(3, 1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

=================================================================
Total params: 181
Trainable params: 181
Non-trainable params: 0
_________________________________________________________________

파라미터 갯수 = feature * units + units * units + bias * units 
            = 1*10 + 10*10 + 1*10 = 120 
            = units * (feaure + units + bias) 
            = 10 * (1 + 10 + 1)
"""

"""
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0
_________________________________________________________________
파라미터 갯수  = rnn params * 4
4배인 이유 : 3개의 게이트(Forget/Input/Output)와 1개의 후보 메모리 셀 사용
-> 그만큼 느리다
"""

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 gru (GRU)                   (None, 10)                390

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

=================================================================
Total params: 451
Trainable params: 451
Non-trainable params: 0
_________________________________________________________________ 
파라미터 갯수  = rnn params * 3 (+약간의 오차)
tensorflow/keras가 390이 나오는 이유 :  bias가 2개를 써서 bias(1x30)가 2배로 나오기때문
pytorch는 360나옴
"""