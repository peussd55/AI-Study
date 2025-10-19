### <<26>>
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout # LSTM 추가
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
x_train_load = np.load(np_path + "keras44_01_x_train.npy")
y_train_load = np.load(np_path + "keras44_01_y_train.npy")
submission_x_load = np.load(np_path + 'keras44_01_x_submission.npy')

x_train_orig, x_test_orig, y_train, y_test = train_test_split(
    x_train_load, y_train_load, test_size=0.3, random_state=8282
)

# LSTM 입력을 위해 데이터 Flatten
num_features = x_train_orig.shape[1] * x_train_orig.shape[2] * x_train_orig.shape[3]
x_train_flat = x_train_orig.reshape(x_train_orig.shape[0], num_features)
x_test_flat = x_test_orig.reshape(x_test_orig.shape[0], num_features)
submission_x_flat = submission_x_load.reshape(submission_x_load.shape[0], num_features)

# 요청사항 1: LSTM 입력을 위한 데이터 Reshape
x_train_reshaped = x_train_flat.reshape(x_train_flat.shape[0], num_features, 1)
x_test_reshaped = x_test_flat.reshape(x_test_flat.shape[0], num_features, 1)
submission_x_reshaped = submission_x_flat.reshape(submission_x_flat.shape[0], num_features, 1)

# 요청사항 2: 모델의 첫 번째 레이어를 LSTM으로 변경
model = Sequential()
# input_shape의 특성 수가 매우 크므로 LSTM 유닛 수를 줄여서 메모리 관리
model.add(LSTM(64, input_shape=(num_features, 1))) 
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

start_time = time.time()
model.fit(
    x_train_reshaped, y_train,
    batch_size=32, epochs=200, verbose=1,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

# 4. 평가, 예측
print("걸린 시간 :", round(end_time - start_time, 2), "초")
results = model.evaluate(x_test_reshaped, y_test)
y_pred = model.predict(x_test_reshaped)
local_logloss_score = log_loss(y_test, y_pred)

print(f'loss: {results[0]}, acc: {results[1]}')
print(f"Log Loss: {local_logloss_score}")
