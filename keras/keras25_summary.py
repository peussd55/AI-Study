### <<12>>

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))

model.summary()
# Param : 연산량(Output*node + bias) (ex : (h0 = W^0 + W1 + W2 + b) + (h1 = W^3 + W4 + W5 + b))
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 4)                 12

#  dense_3 (Dense)             (None, 1)                 5

# =================================================================
# Total params: 31
# Trainable params: 31      # 남의 모델가져와서 일부분만 frozen하고 재학습돌릴때 Trainable params갯수만큼 훈련할 수있다.
# Non-trainable params: 0
# _________________________________________________________________