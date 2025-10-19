### <<26>>

# 30-4카피

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout  # Conv1D 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드 및 전처리
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# 결측치 처리
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=999
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
    
    # Conv1D 입력을 위한 3D 변환 (samples, timesteps, features)
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 9, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 9, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델 구성 (Conv1D 사용)
    model = Sequential()
    model.add(Conv1D(128, kernel_size=2, activation='relu', input_shape=(9, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # 회귀 출력층

    # 3. 컴파일 및 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
        restore_best_weights=True
    )

    hist = model.fit(
        x_train_reshaped, y_train,
        epochs=1000,
        batch_size=32,
        verbose=3,
        validation_split=0.1,
        callbacks=[es]
    )

    # 4. 평가 및 예측
    loss = model.evaluate(x_test_reshaped, y_test)
    y_pred = model.predict(x_test_reshaped)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print('r2 스코어:', r2)
    print('RMSE:', rmse)
    print('loss:', loss[0])

    # 제출용 예측
    test_csv_scaled = scaler.transform(test_csv) if scaler else test_csv.values
    test_csv_reshaped = test_csv_scaled.reshape(test_csv_scaled.shape[0], 9, 1)
    y_submit = model.predict(test_csv_reshaped)
    submission_csv['count'] = y_submit
    
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{scaler_name}_{current_time}.csv')
