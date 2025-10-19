### <<25>>

# Bidirectional : 양방향 시계열훈련

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

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
# Bidirectional은 모델은 아니고 시계열훈련을 양방향으로 하겠다는 레이어임(params 2배)
model = Sequential()
# model.add(LSTM(units=10, input_shape=(3,1)))
model.add(Bidirectional(GRU(units=10), input_shape=(3,1)))  
# Bidirectional는 시계열레이어를 왔다갔다하기때문에 즉, 시계열데이터를 2번 학습하기때문에 시계열레이어를 두번쌓는것과 달리 성능이 좋아질 여지가 있다.
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# Bidirectional(LSTM)
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  bidirectional (Bidirectiona  (None, 20)               960 = (((1+10)*10 + 1*10)*4)*2)
#  l)

#  dense (Dense)               (None, 7)                 147

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 1,115
# Trainable params: 1,115
# Non-trainable params: 0
# _________________________________________________________________

###### param 개수 ######
"""
RNN : (10*10 + 1*10 + 10 = 120) + (10*7+7=77) + (1*8) = 205
Bidirectional : 240 + 20*7+7 + 8*1 = 395

GRU : (120*3 + 10*3)*3 + 77 + 1*8 = 475
Bidirectional : 390*2=780 + 20*7+7 + 1*8 = 935

LSTM : 480 + 10*7+7 + 8*1
Bidirectional : ((((1+10)*10 + 1*10)*4)*2) = 960) + 20*7+7 + 8*1
    
"""