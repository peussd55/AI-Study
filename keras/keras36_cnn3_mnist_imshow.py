### <<17>>

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train)
# [[[0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   ...
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]]
#    ...
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]]]
print(x_train[0])   # 학습데이터는 2차원으로 된 이미지 데이터임
# [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#   ...
#   [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136
#   175  26 166 255 247 127   0   0   0   0]
#   [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253
#   225 172 253 242 195  64   0   0   0   0]
#   ...
#    [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]]
print(y_train[0])   # 5

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) : (60000, 28, 28, 1)이 아닌 형태로 제공하는 이유? -> 흑백데이터는 2차원으로 충분히 표현할수있기때문에. 그러나 모델을 구성할땐 채널이추가된 형태로 reshape해야한다.
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))   
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
# y가 균형데이터 -> 평가지표 accuracy써도됨
print(pd.value_counts(y_test))  # 테스트데이터는 불균형해도 상관없음

aaa = 42
print(y_train[aaa])
import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
plt.imshow(x_train[aaa], 'rainbow_r')
plt.show()