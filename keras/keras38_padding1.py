### <<18>>

# [실습]
# 100,100,3 이미지를
# 10,10,11으로 줄여봐

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

# 2. 모델구성
model = Sequential()
model.add(Conv2D(1, (11,11), input_shape=(100,100,3),
                 strides=1,
                 #  padding='same',
                 padding='valid',
                 ))
model.add(Conv2D(8, 11))
model.add(Conv2D(8, 11))
model.add(Conv2D(8, 11))
model.add(Conv2D(8, 11))
model.add(Conv2D(8, 11))
model.add(Conv2D(8, 11))
model.add(Conv2D(8, 11))
model.add(Conv2D(11, 11))
model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 90, 90, 1)         364

 conv2d_1 (Conv2D)           (None, 80, 80, 8)         976

 conv2d_2 (Conv2D)           (None, 70, 70, 8)         7752

 conv2d_3 (Conv2D)           (None, 60, 60, 8)         7752

 conv2d_4 (Conv2D)           (None, 50, 50, 8)         7752

 conv2d_5 (Conv2D)           (None, 40, 40, 8)         7752

 conv2d_6 (Conv2D)           (None, 30, 30, 8)         7752

 conv2d_7 (Conv2D)           (None, 20, 20, 8)         7752

 conv2d_8 (Conv2D)           (None, 10, 10, 11)        10659

=================================================================
"""
model1 = Sequential()
model1.add(Conv2D(1, (3,3), input_shape=(100,100,3),
                 strides=3,
                 #  padding='same',
                 padding='valid',
                 ))
model1.add(Conv2D(11, 3, 3))
model1.add(Conv2D(11, 2, 1)
           )
model1.summary()    # 98 -> 49 -> 33
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_9 (Conv2D)           (None, 33, 33, 1)         28

 conv2d_10 (Conv2D)          (None, 11, 11, 11)        110       

 conv2d_11 (Conv2D)          (None, 10, 10, 11)        495

=================================================================
Total params: 633
Trainable params: 633
Non-trainable params: 0
_________________________________________________________________
"""