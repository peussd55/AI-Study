### <<23>>

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array(range(1,101))  # 1차원
x_predict = np.array(range(96,107)) # 101 ~ 107을 찾기
print(x_predict)        # [ 96  97  98  99 100 101 102 103 104 105 106]
print(x_predict.shape)  # (11,)

timesteps = 6

# x.shape = n,5,1

# 모델까지 만들기

timesteps = 6
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
# [[  1   2   3   4   5   6]    
#     ...
#  [ 95  96  97  98  99 100]]
print(bbb.shape)   # (95, 6)

x = bbb[:, :-1]     # [행, 열]. 모든 행, 마지막 열 제외
y = bbb[:, -1]    # (배치(블록), 행, 열) 에서 첫번째 행만 선택 그리고 두번째 열만 선택
print(x)
# [[ 1  2  3  4  5]
#     ...
#   [95 96 97 98 99]]
print('x.shape:', x.shape)  # (95, 5)

print(y)
# [  6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
#   24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41
#   42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59
#   60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77
#   78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
#   96  97  98  99 100]
print('y.shape:',y.shape)  # (95,)

# x 3차원으로 변환
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x)
# [[[ 1]
#   [ 2]
#   [ 3]
#   [ 4]
#   [ 5]]
# ...
# [[95]
#   [96]
#   [97]
#   [98]
#   [99]]]
print('x.shape:',x.shape)  # (95, 5, 1)

# x_predict도 reshape
x_predict = split_x(x_predict, timesteps=5)
print(x_predict)
# 행이 7개라서 예측값 y_pred도 7개 나옴
# [[ 96  97  98  99 100]
#  [ 97  98  99 100 101]
#  [ 98  99 100 101 102]
#  [ 99 100 101 102 103]
#  [100 101 102 103 104]
#  [101 102 103 104 105]
#  [102 103 104 105 106]]

print('x_predict.shape:',x_predict.shape)  # (7, 5)

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=512, input_length=5, input_dim=1, activation='relu'))
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
    patience=30,              
    restore_best_weights=True, 
)

model.fit(x, y, epochs=200, callbacks=[es],)

path = './_save/keras55/'
model.save_weights(path + 'keras55_3_LSTM_save.h5')

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
y_pred = model.predict(x_predict)

print('x의 결과 : ', y_pred)
# [[101.0035  ]
#  [102.006584]
#  [103.00999 ]
#  [104.0136  ]
#  [105.01744 ]
#  [106.02151 ]
#  [107.02577 ]]

