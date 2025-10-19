### <<23>>

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array([
            [1,2,3,4,5,6,7,8,9,10],
            [9,8,7,6,5,4,3,2,1,0]]).T
print(a.shape)  # (10, 2)
print(a)
# [[ 1  9]
#  [ 2  8]
#  [ 3  7]
#  [ 4  6]
#  [ 5  5]
#  [ 6  4]
#  [ 7  3]
#  [ 8  2]
#  [ 9  1]
#  [10  0]]

# (참고) 2번째 열(인덱스 1) 값을 복사하여 새로운 열 생성
# new_col = a[:, 1]       # a에서 두번째 열(첫번째 열은 0임)을 선택 해서 1차원배열 반환.
# print(new_col)          # [9 8 7 6 5 4 3 2 1 0]
# print(new_col.shape)    # (10,)
# new_col = a[:, 1].reshape(-1,1)
# print(new_col.shape)    # (10,1)


# 10행 2열의 데이터중에 
# 첫번째와 두번째 컬럼을 x로 잡고,
# 두번째 컬럼을 y값으로 잡는다.
timesteps = 5
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
# [[[ 1  9]
#   [ 2  8]
#   [ 3  7]
#   [ 4  6]
#   [ 5  5]]

#  [[ 2  8]
#   [ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]]

#  [[ 3  7]
#   [ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]]

#  [[ 4  6]
#   [ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]]

#  [[ 5  5]
#   [ 6  4]
#   [ 7  3]
#   [ 8  2]
#   [ 9  1]]

#  [[ 6  4]
#   [ 7  3]
#   [ 8  2]
#   [ 9  1]
#   [10  0]]]
print(bbb.shape)    # (6, 5, 2)

x = bbb[:, :-1, :]  # (배치(블록), 행, 열) 에서 마지막행만 제거
y = bbb[:, 0, 1]    # (배치(블록), 행, 열) 에서 첫번째 행만 선택 그리고 두번째 열만 선택
print(x)
# [[[1 9]
#   [2 8]
#   [3 7]
#   [4 6]]

#  [[2 8]
#   [3 7]
#   [4 6]
#   [5 5]]

#  [[3 7]
#   [4 6]
#   [5 5]
#   [6 4]]

#  [[4 6]
#   [5 5]
#   [6 4]
#   [7 3]]

#  [[5 5]
#   [6 4]
#   [7 3]
#   [8 2]]

#  [[6 4]
#   [7 3]
#   [8 2]
#   [9 1]]]
print(y)    # [9 8 7 6 5 4]

print(x.shape, y.shape) # (6, 4, 2) (6,)
# (batch_size, timesteps, feature)

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=512, input_length=4, input_dim=2, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping( 
    monitor = 'loss',      
    mode = 'min',              
    patience=600,              
    restore_best_weights=True, 
)

model.fit(x, y, epochs=2000, callbacks=[es],)

path = './_save/keras55/'
model.save_weights(path + 'keras55_2_LSTM_save.h5')

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

y_pred = model.predict(x)

print('x의 결과 : ', y_pred)
# [[9.000074 ]
#  [7.9998107]
#  [7.0001597]
#  [5.999966 ]
#  [4.9999886]
#  [4.000005 ]]
