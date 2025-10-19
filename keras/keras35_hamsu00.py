### <<16>>

# 07-2-1 카피

import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9,8,7,6,5,4,3,2,1,0]])    # (3, 10) : 잘못된 데이터

y = np.array([1,2,3,4,5,6,7,8,9,10])   
x = np.transpose(x)                     # (3, 10) -> (10, 3) / y가 10개라서 데이터갯수(행)도 10개여야한다.

print(x.shape)
print(y.shape)

# 2-1. 모델구성(순차형)
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(9, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Dense(1))
model.summary()
# _________________________________________________________________
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dropout (Dropout)           (None, 9)                 0

#  dense_2 (Dense)             (None, 8)                 80

#  dropout_1 (Dropout)         (None, 8)                 0

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# dropout해도 total parmas이 그대로인 이유 : 학습할때마다 활성화되는 params가 바뀌기 때문에 모든 param이 학습에 참여한다고 볼 수있기때문이다. 

# 2-2. 모델구성 (함수형) : 모델정의를 뒤에한다.
input1 = Input(shape=(3,))
dense1 = Dense(10, name='ys1')(input1)  # name : 레이어의 이름 지정
dense2 = Dense(9, name='ys2', activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(8)(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(7)(drop2)
output1 = Dense(1)(dense4)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()
# _________________________________________________________________
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  ys1 (Dense)                 (None, 10)                40

#  ys2 (Dense)                 (None, 9)                 99

#  dropout_2 (Dropout)         (None, 9)                 0

#  dense_5 (Dense)             (None, 8)                 80

#  dropout_3 (Dropout)         (None, 8)                 0

#  dense_6 (Dense)             (None, 7)                 63

#  dense_7 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________
