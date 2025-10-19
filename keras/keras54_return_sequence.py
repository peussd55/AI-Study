### <<23>>

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

# 1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]
            ])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]).reshape(1, 3, 1)    # (3,) -> (1,3,1)

x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=256, input_shape=(3,1), return_sequences=True, activation='relu'))    # return_sequences=True : 출력의 차원을 유지함(3차원)
model.add(GRU(units=128, input_shape=(3,1), return_sequences=True, activation='relu'))      # 입력이 3차원이라 시계열 레이어 붙일수있다.
model.add(SimpleRNN(units=64, input_shape=(3,1), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 3, 512)            1052672
 gru (GRU)                   (None, 3, 256)            591360
 simple_rnn (SimpleRNN)      (None, 128)               49280

 dense (Dense)               (None, 64)                8256

 dense_1 (Dense)             (None, 32)                2080

 dense_2 (Dense)             (None, 16)                528

 dense_3 (Dense)             (None, 1)                 17

=================================================================
"""
# exit()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = './_save/keras54/keras54_LSTM_save.h5'

es = EarlyStopping( 
    monitor = 'loss',      
    mode = 'min',              
    patience=600,              
    restore_best_weights=True, 
)

model.fit(x, y, epochs=2000, callbacks=[es],)

path = './_save/keras53/'
# model.save(path + 'keras26_3_save.h5')  # 학습가중치 저장
model.save_weights(path + 'keras54_LSTM_save.h5')

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

y_pred = model.predict(x_predict)

print('[50,60,70]의 결과 : ', y_pred)
# [50,60,70]의 결과 :  [[80.4497]]