### <<26>>
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout # LSTM 추가
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

# 원핫 인코딩
y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=4325, stratify=y
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=5234, stratify=y_train
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
]

original_x_train = x_train.copy()
original_x_test = x_test.copy()
original_x_val = x_val.copy()

for scaler_name, scaler in scalers:
    print(f"\n\n======= 스케일러: {scaler_name} =======")
    x_train = original_x_train.copy()
    x_test = original_x_test.copy()
    x_val = original_x_val.copy()

    if scaler is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)

    # 요청사항 1: LSTM 입력을 위한 데이터 Reshape
    # 2D 데이터를 LSTM 모델에 입력하기 위해 3D 텐서로 변환합니다. (샘플 수, 타임스텝, 특성 수)
    # 여기서는 64개의 특성을 (64, 1) 형태로 변환합니다.
    x_train_reshaped = x_train.reshape(x_train.shape[0], 64, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 64, 1)
    x_val_reshaped = x_val.reshape(x_val.shape[0], 64, 1)

    # 요청사항 2: 모델의 첫 번째 레이어를 LSTM으로 변경
    model = Sequential()
    model.add(LSTM(128, input_shape=(64, 1))) # input_shape는 (타임스텝, 특성 수)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax')) # 클래스 10개

    # 3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
    
    start_time = time.time()
    model.fit(
        x_train_reshaped, y_train,
        epochs=200, batch_size=32, verbose=0,
        validation_data=(x_val_reshaped, y_val),
        callbacks=[es]
    )
    end_time = time.time()

    # 4. 평가, 예측
    print("걸린 시간 :", round(end_time - start_time, 2), "초")
    results = model.evaluate(x_test_reshaped, y_test)
    y_pred = np.argmax(model.predict(x_test_reshaped), axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_test_argmax, y_pred)
    f1 = f1_score(y_test_argmax, y_pred, average='macro')
    
    print(f"loss: {results[0]}, acc: {results[1]}")
    print(f"Accuracy Score: {acc}, F1-Score: {f1}")
