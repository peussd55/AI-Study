### <<54>>

import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import tensorflow as tf 
import random 

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)
print(tf.__version__)   # 2.7.4

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0
# _________________________________________________________________

print(model.weights)    # 가중치
"""
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>,   
  -> 초기가중치 : 1x3 이므로 3개
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
  -> 초기바이어스 : 3개

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, 
  -> 두번째 층 가중치 : 3x2 이므로 6개
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
  -> 두번째 바이어스 : 2개

<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>,
  -> 세번째 층 가중치 : 2x1 이므로 2개
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
  -> 새번째 바이어스 : 1개
"""
# :: 케라스는 input이 벡터(1차원)이면 첫번째 히든레이어와 곱이 가능한 형태로 reshape한다. intput : (n,) -> (n,1) * (1,3) -> (n,3)
# :: 인접한 레이어들간은 행렬곱이 가능한 shape : ex : (n,3) x (3,2) -> (n,2) 앞 행렬의 열과 뒤 행렬의 행이 같아야 연산가능
# :: 곱해진 레이어들의 shape는 행렬곱의 결과 : ex : (n,2) x (2,1) -> (n,1) shape로 나옴

print("==========================================")
print(model.trainable_weights)  # 훈련가능한 가중치
"""
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.92561173,  0.8256177 ],
       [ 0.6200088 ,  1.0182774 ],
       [-0.5191052 , -0.6304303 ]], dtype=float32)>, 
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,    
           
<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.02628279],
       [-1.074922  ]], dtype=float32)>, 
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
"""
print("==========================================")

print(len(model.weights))           # 6 : 덩어리(layer / 가중치+노드 / a11w11+a12w12... + a21w21... + ...) 3개
print(len(model.trainable_weights)) # 6

# 전이학습 : 남이만든 모델을가져와서 동결해서 그대로쓰거나 재학습하는 행위를 포괄하는 과정
# 동결, freezen : 모델을 가져와서 재학습하지 않는다(가중치 그대로사용)

##################### 동결 #####################
model.trainable = False
##################### 동결 #####################
print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0 : 동결했으므로 훈련할수있는 가중치가 0개
# non-trainable = 역전파계산을 하지 않는다.

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 3)                 6

#  dense_1 (Dense)             (None, 2)                 8

#  dense_2 (Dense)             (None, 1)                 3

# =================================================================
# Total params: 17
# Trainable params: 0 : 학습가능한 파라미터 0개
# Non-trainable params: 17
# _________________________________________________________________

print("==========================================")
print(model.weights)
# array([[-0.92561173,  0.8256177 ],
#        [ 0.6200088 ,  1.0182774 ],
#        [-0.5191052 , -0.6304303 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=     
# array([[-0.02628279],
#        [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
print("==========================================")
print(model.trainable_weights)  
# []