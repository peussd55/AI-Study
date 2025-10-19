## <<26>>

# 30-7 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM  # LSTM 추가
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 데이터 전처리
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
x = x.replace(0, np.nan).fillna(x.median())  # 0값을 NaN으로 변환 후 중앙값으로 대체

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=69758282, shuffle=True
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
    model.add(LSTM(64, input_shape=(8, 1)))  # 첫 번째 레이어를 LSTM으로 변경
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류 출력층

    # 3. 컴파일 및 훈련
    model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['acc'])

    es = EarlyStopping( 
        monitor='val_loss',       
        mode='min',               
        patience=115,             
        restore_best_weights=True,  
    )

    start_time = time.time()
    hist = model.fit(
        x_train_reshaped, y_train,
        epochs=1000, 
        batch_size=2,
        verbose=3, 
        validation_split=0.15,
        callbacks=[es]
    )
    end_time = time.time()

    # 4. 평가 및 예측
    results = model.evaluate(x_test_reshaped, y_test)
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print("loss :", results[0])
    print("acc :", results[1])
    
    y_pred = model.predict(x_test_reshaped)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, np.round(y_pred))
    print("accuracy_score :", accuracy)

    # 제출용 예측
    test_csv_scaled = scaler.transform(test_csv) if scaler else test_csv.values
    test_csv_reshaped = test_csv_scaled.reshape(test_csv_scaled.shape[0], 8, 1)
    y_submit = model.predict(test_csv_reshaped)
    submission_csv['Outcome'] = np.round(y_submit)
    
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{scaler_name}_{current_time}.csv', index=False)
