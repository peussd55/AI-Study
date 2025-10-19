### <<26>>

# 30-3카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout  # Conv1D 추가
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np 

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1111
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
    # 10개 피처를 timesteps=10, features=1로 설정
    x_train_reshaped = x_train_scaled.reshape(x_train_scaled.shape[0], 10, 1)
    x_test_reshaped = x_test_scaled.reshape(x_test_scaled.shape[0], 10, 1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('x_train_reshaped shape:', x_train_reshaped.shape)
    print('x_test_reshaped shape:', x_test_reshaped.shape)

    # 2. 모델 구성 (Conv1D 사용)
    model = Sequential()
    model.add(Conv1D(128, kernel_size=2, activation='relu', input_shape=(10, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # 회귀 출력층 (활성화 함수 없음)

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
        validation_split=0.2,
        callbacks=[es]
    )

    # 4. 평가 및 예측
    loss = model.evaluate(x_test_reshaped, y_test)
    results = model.predict(x_test_reshaped)
    
    r2 = r2_score(y_test, results)
    print(f"=== 현재 적용 스케일러: {scaler_name} ===")
    print('r2 스코어 :', r2)
