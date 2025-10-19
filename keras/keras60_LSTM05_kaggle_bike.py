### <<26>>

# 30-5 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM  # LSTM 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터 로드 및 전처리
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
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
    
    # LSTM 입력을 위한 3D 변환 (samples, timesteps, features)
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 8, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 8, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델 구성 (LSTM 사용)
    model = Sequential()
    model.add(LSTM(128, input_shape=(8, 1)))  # 첫 번째 레이어를 LSTM으로 변경
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))  # 출력 레이어

    # 3. 컴파일 및 훈련
    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
        restore_best_weights=True
    )

    hist = model.fit(
        x_train_reshaped, y_train,
        epochs=1000,
        batch_size=24,
        verbose=3,
        validation_split=0.1,
        callbacks=[es]
    )

    # 4. 평가 및 예측
    loss = model.evaluate(x_test_reshaped, y_test)
    results = model.predict(x_test_reshaped)
    
    r2 = r2_score(y_test, results)
    rmse = np.sqrt(mean_squared_error(y_test, results))
    
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print('loss 값:', loss)
    print('r2 스코어:', r2)
    print('rmse 스코어:', rmse)

    # 제출용 예측 생성
    test_csv_scaled = scaler.transform(test_csv) if scaler else test_csv.values
    test_csv_reshaped = test_csv_scaled.reshape(test_csv_scaled.shape[0], 8, 1)
    y_submit = model.predict(test_csv_reshaped)
    
    # 제출 파일 저장
    submission_csv['count'] = y_submit
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{scaler_name}_{current_time}.csv', index=False)
