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
x_predict = np.array([50,60,70])

# x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=512, input_length=3, input_dim=1, activation='relu'))
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

path = './_save/keras53/'
# model.save(path + 'keras26_3_save.h5')  # 학습가중치 저장
model.save_weights(path + 'keras53_LSTM_save.h5')

# 4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_predict = np.array([50,60,70]).reshape(1, 3, 1)    # (3,) -> (1,3,1)
y_pred = model.predict(x_predict)

print('[50,60,70]의 결과 : ', y_pred)
# [50,60,70]의 결과 :  [[80.00062]]