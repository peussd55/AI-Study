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
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

###################################### 동결 ######################################
# model.trainable = False   # 동결
model.trainable = True    # 동결X : default

print("=====================================================") 
print(model.weights)  # 초기 가중치. bias는 0
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 

# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.92561173,  0.8256177 ],
#        [ 0.6200088 ,  1.0182774 ],
#        [-0.5191052 , -0.6304303 ]], dtype=float32)>, 
# <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 

# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.02628279],
#        [-1.074922  ]], dtype=float32)>, 
# <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
print("=====================================================")

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

# 4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)
# 동결 False : 결과잘나옴
# [[1.0000001]
#  [1.9999999]
#  [3.0000002]
#  [4.       ]
#  [4.9999995]]

# 동결 True : 역전파계산없이 초기가중치로만 예측해서 결과 잘 안나옴. loss가 매 epochs마다 그대로임.
# [[0.45656443]
#  [0.91312885]
#  [1.369693  ]
#  [1.8262577 ]
#  [2.282822  ]]

# x=1, y=1, 가중치 동결 후 훈련.
# [[0.45656443]]
# 손계산 : 약 0.439

# x=1, y=1, 가중치 열어놓고 훈련.
# [[1.]]
print("=====================================================") 
print(model.weights)  # 갱신된 가중치
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.08451442, -0.08841534,  0.82829833]], dtype=float32)>, 
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([-0.05151878, -0.05361512,  0.0539354 ], dtype=float32)>, 
# 
# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.9780188,  0.7826859],
#        [ 0.6975388,  1.0858246],
#        [-0.5847606, -0.6844676]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.06426802, -0.05272117], dtype=float32)>, 
# 
# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[-0.08096293],
#        [-1.1340269 ]], dtype=float32)>, 
# <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.05221002], dtype=float32)>]
print("=====================================================") 