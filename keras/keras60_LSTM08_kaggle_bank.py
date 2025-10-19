### <<26>>

# 30-8 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM  # LSTM 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드 및 전처리
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 결측치 확인
print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

# 문자 데이터 수치화(인코딩)
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# CustomerId, Surname 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=8282, shuffle=True
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
        x_train_scaled = x_train.values
        x_test_scaled = x_test.values
    else:
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    
    # LSTM 입력을 위한 3D 변환 (samples, timesteps, features)
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 10, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 10, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델 구성 (LSTM 사용)
    model = Sequential()
    model.add(LSTM(128, input_shape=(10, 1)))  # 첫 번째 레이어를 LSTM으로 변경
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류 출력층

    # 3. 컴파일 및 훈련
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc', 'AUC']
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
        batch_size=256,
        verbose=3,
        validation_split=0.3,
        callbacks=[es]
    )
    end_time = time.time()

    # 4. 평가 및 예측
    results = model.evaluate(x_test_reshaped, y_test)
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print("loss :", results[0])
    print("acc :", results[1])
    
    y_pred = model.predict(x_test_reshaped)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy :", accuracy)

    # 제출용 예측
    test_csv_scaled = scaler.transform(test_csv) if scaler else test_csv.values
    test_csv_reshaped = test_csv_scaled.reshape(test_csv_scaled.shape[0], 10, 1)
    y_submit = model.predict(test_csv_reshaped)
    submission_csv['Exited'] = y_submit
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{scaler_name}_{current_time}.csv', index=False)
