### <<23>>

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array(range(1, 11))
# a = a.reshape(a.shape[0], 1)    # (10, 1) // len 10
# [[ 1]
#  [ 2]
#  [ 3]
#  [ 4]
#  [ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]
# a = a.reshape(a.shape[0], -1)   # (10, 1) // len 10
# [[ 1]
#  [ 2]
#  [ 3]
#  [ 4]
#  [ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]
# a = a.reshape(1, a.shape[0])    # (1, 10) // [[ 1  2  3  4  5  6  7  8  9 10]] // len 1
# a = a.reshape(-1, a.shape[0])     # (1, 10) // [[ 1  2  3  4  5  6  7  8  9 10]] // len 1
timesteps = 5

print(a.shape)  # (10,)

# (중요) ★★★ len함수는 axis=0을 기준으로 길이를 잡는다 ★★★

def split_x(dataset, timesteps):
    print('dataset:',dataset)
    print('timesteps:',timesteps)
    aaa = []
    print('len(dataset):',len(dataset))
    print('timesteps + 1:',timesteps + 1)
    for i in range(len(dataset) - timesteps + 1):   # for i in range(6)
        print('len(dataset) - timesteps + 1:', (len(dataset) - timesteps + 1))
        subset = dataset[i : i+timesteps]
        print('i:',i)
        print('i+timesteps:',i+timesteps)
        print('subset:',subset)
        aaa.append(subset)
        print('aaa:',aaa)
    return np.array(aaa)

bbb = split_x(dataset=a, timesteps=timesteps)
print(bbb)
print(bbb.shape)    # (6, 5)

x = bbb[:, :-1] # [행, 열]. 모든 행, 마지막 열 제외
y = bbb[:, -1]  # [행, 열]. 모든 행, 마지막 열 만
print(x)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]
print(y)
# [ 5  6  7  8  9 10]
print(x.shape, y.shape) # (6, 4) (6,)

"""
★★★ [len함수는 axis=0을 기준으로 길이를 잡는다] ★★★

1차원: axis=0(원소)

2차원: axis=0(행), axis=1(열)

3차원: axis=0(면/샘플/깊이), axis=1(행), axis=2(열)

4차원: axis=0(샘플), axis=1(채널/깊이), axis=2(행), axis=3(열)
"""