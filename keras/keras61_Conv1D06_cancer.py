### <<26>>

# 30-6카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization  # Conv1D 추가
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 전처리
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=8282, shuffle=True
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
]

for scaler_name, scaler in scalers:
    # 데이터 스케일링
    if scaler is None:
        x_train_scaled = x_train
        x_test_scaled = x_test
    else:
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    
    # Conv1D 입력을 위한 3D 변환 (samples, timesteps, features)
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 30, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 30, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델 구성 (Conv1D 사용)
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(30, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류 출력층

    # 3. 컴파일 및 훈련
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )
    
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
        restore_best_weights=True
    )

    start_time = time.time()
    hist = model.fit(
        x_train_reshaped, y_train,
        epochs=1000,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[es]
    )
    end_time = time.time()

    # 4. 평가 및 예측
    results = model.evaluate(x_test_reshaped, y_test)
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print("loss :", round(results[0], 4))
    print("acc :", round(results[1], 4))
    
    y_pred = model.predict(x_test_reshaped)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy_score :", accuracy)
    print("걸린시간 :", round(end_time - start_time, 2), '초')
