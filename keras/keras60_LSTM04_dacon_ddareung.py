### <<26>>

# 30-4 카피

import numpy as np 
import pandas as pd 
print(np.__version__)   # 1.23.0
print(pd.__version__)   # 2.2.3

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, LSTM  # LSTM 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정
test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# 결측치 처리
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)   # count 컬럼 제거
y = train_csv['count']                  # 타겟 변수

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2, 
    random_state=999
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
]

for scaler_name, scaler in scalers:
    # 데이터 스케일링 or 원본 데이터 사용
    if scaler is None:
        x_train_scaled = x_train.values
        x_test_scaled = x_test.values
    else:
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    
    # LSTM 입력을 위한 3D 변환 (samples, timesteps, features)
    # timesteps=9, features=1로 설정
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 9, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 9, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델구성 (LSTM으로 변경)
    model = Sequential()
    model.add(LSTM(128, input_shape=(9, 1)))  # 첫 번째 레이어를 LSTM으로 변경
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # 3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=100,
        restore_best_weights=True
    )

    hist = model.fit(x_train_reshaped, y_train,
                    epochs=1000,
                    batch_size=24,
                    verbose=3,
                    validation_split=0.1,
                    callbacks=[es])

    # 4. 평가, 예측
    loss = model.evaluate(x_test_reshaped, y_test)
    results = model.predict(x_test_reshaped)
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, results)
    print('r2 스코어 :', r2)
    
    # submission.csv 생성
    test_csv_scaled = scaler.transform(test_csv) if scaler else test_csv.values
    test_csv_reshaped = test_csv_scaled.reshape(test_csv_scaled.shape[0], 9, 1)
    y_submit = model.predict(test_csv_reshaped)
    submission_csv['count'] = y_submit
    
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    submission_csv.to_csv(f'{path}submission_{scaler_name}_{current_time}.csv')
