### <<18>>

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

# 2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=1,
                 padding='same',    # padding='same' : 출력의 크기를 shape/strides(소수점올림) 만큼 맞추기위해 제로패딩을 부여
                 #  padding='valid',
                 ))
model.add(Conv2D(filters=9, kernel_size=(3,3), 
                 strides=1,
                 padding='valid',   #디폴트
                 ))
model.add(Conv2D(8, 4)) # filters=8, kernel_size=(4,4)

model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 10, 10, 10)        50        

 conv2d_1 (Conv2D)           (None, 8, 8, 9)           819       

 conv2d_2 (Conv2D)           (None, 5, 5, 8)           1160      

=================================================================
"""
model1 = Sequential()
model1.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=2,
                  padding='same',    # padding='same' : 출력의 크기를 shape/strides(소수점올림) 만큼 맞추기위해 제로패딩을 부여
                #   padding='valid',
                 )) # Output Shape가 same일때나 valid일때나 동일 :(None, 5, 5, 10)
model1.add(Conv2D(8, 4)) # filters=8, kernel_size=(4,4)
model1.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_3 (Conv2D)           (None, 5, 5, 10)          50

 conv2d_4 (Conv2D)           (None, 2, 2, 8)           1288

=================================================================    
"""

model1 = Sequential()
model1.add(Conv2D(10, (2,2), input_shape=(10,8,1), strides=2, padding='same',)) 
model1.summary()    
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_5 (Conv2D)           (None, 5, 4, 10)          50

=================================================================
"""

model1 = Sequential()
model1.add(Conv2D(10, (2,2), input_shape=(10,7,1), strides=2, padding='valid',)) 
model1.summary() # 너비는 7/2=3.5로 소실되는 칸 생김
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_6 (Conv2D)           (None, 5, 3, 10)          50

=================================================================
"""

model1 = Sequential()
model1.add(Conv2D(10, (2,2), input_shape=(10,7,1), strides=2, padding='same',)) 
model1.summary() # 소실되는 칸이 안생기게 너비방향으로 패딩부여(소수점올림) : 3.5 -> 4로 맞추기 위해 패딩부여
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_7 (Conv2D)           (None, 5, 4, 10)          50

=================================================================
"""