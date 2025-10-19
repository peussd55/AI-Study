import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# --- 1. 데이터 로드 및 전처리 ---
print("1. 데이터 로드 및 전처리 시작...")

# 데이터 로드 경로 (***사용자 환경에 맞게 조정 필수***)
data_path = 'C:/Study25/_data/kaggle/jena/jena_climate_2009_2016.csv'
df = pd.read_csv(data_path)

# 'Date Time' 컬럼을 datetime 형식으로 변환하고 인덱스로 설정
df['Date Time'] = pd.to_datetime(df['Date Time'])
df = df.set_index('Date Time')

# 사용할 특성 컬럼 정의
# 예측 목표는 'wd (deg)' 이지만, 입력에는 관련 있는 모든 특성을 사용합니다.
feature_cols = ['wd (deg)', 'wv (m/s)', 'max. wv (m/s)'] # 풍향, 풍속, 최대 풍속

# NaN/Inf 값 확인 및 처리 (모든 feature_cols에 대해)
for col in feature_cols:
    if col == 'wd (deg)':
        df[col] = df[col].replace(999.9, np.nan) # 풍향의 특수 결측치 (999.9)를 NaN으로 변환
    
    # 일반적인 NaN 값 처리: 이전 값으로 채우기 (시계열 데이터에 적합)
    # 혹시 맨 앞에 NaN이 있을 경우를 대비하여 bfill도 추가
    print(f"처리 전 '{col}' 컬럼의 NaN 개수: {df[col].isnull().sum()}")
    df[col] = df[col].fillna(method='ffill')
    df[col] = df[col].fillna(method='bfill') 
    print(f"처리 후 '{col}' 컬럼의 NaN 개수: {df[col].isnull().sum()}")

# 데이터 스케일링
# 선택된 특성 컬럼들을 전체적으로 스케일링합니다.
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[feature_cols])

# 예측 목표인 'wd (deg)' 값만을 위한 별도의 스케일러 생성
# 예측값을 원래 스케일로 되돌릴 때 사용합니다.
wd_scaler = MinMaxScaler()
wd_scaler.fit(df['wd (deg)'].values.reshape(-1, 1))

# timesteps (과거 룩백 기간) 및 output_steps (미래 예측 기간) 정의
timesteps = 144 # 과거 144개 데이터를 보고
output_steps = 144 # 미래 144개 값을 예측
n_features = len(feature_cols) # 사용할 특성(컬럼)의 개수
print('n_features:',n_features) # 3

# 시계열 데이터셋 생성 함수 (다중 특성 및 다중 출력에 맞게 수정)
# X는 (샘플 수, timesteps, 특성 수) 형태
# Y는 (샘플 수, output_steps) 형태 (예측 목표 특성만)
def create_sequences_multi_feature_multi_output(data, timesteps, output_steps, target_feature_idx):
    X, y = [], []
    for i in range(len(data) - timesteps - output_steps + 1):
        # X: 입력 시퀀스 - 모든 특성을 포함합니다.
        subset_x = data[i : (i + timesteps), :] 
        X.append(subset_x)
        
        # Y: 출력 시퀀스 - 예측 목표 특성('wd (deg)')만 가져옵니다.
        subset_y = data[(i + timesteps):(i + timesteps + output_steps), target_feature_idx]
        y.append(subset_y)
    return np.array(X), np.array(y)

# 'wd (deg)' 컬럼이 feature_cols에서 몇 번째 인덱스인지 확인
target_feature_index = feature_cols.index('wd (deg)')

X, y = create_sequences_multi_feature_multi_output(data_scaled, timesteps, output_steps, target_feature_index)

# 훈련 데이터셋 분리 (예측 시작 날짜를 기준으로)
prediction_start_dt = pd.to_datetime('2016-12-31 00:10:00')
# 원본 DataFrame에서 예측 시작 날짜의 인덱스 위치를 찾습니다.
start_idx_original_df = df.index.get_loc(prediction_start_dt)

# 훈련 데이터의 끝점 계산:
# 예측 대상 시퀀스에 사용되는 데이터와 훈련 데이터가 겹치지 않도록 합니다.
# (start_idx_original_df - output_steps)는 첫 예측 y 시퀀스의 시작 인덱스입니다.
# 그 y 시퀀스를 생성하는 X 시퀀스는 (start_idx_original_df - output_steps - timesteps)에서 시작합니다.
# +1은 파이썬 슬라이싱의 특성상 마지막 인덱스를 포함하기 위함입니다.
train_data_end_idx = start_idx_original_df - output_steps - timesteps + 1

X_train, y_train = X[:train_data_end_idx], y[:train_data_end_idx]

print(f"훈련 데이터셋 형태: X_train={X_train.shape}, y_train={y_train.shape}")  # 훈련 데이터셋 형태: X_train=(420120, 144, 3), y_train=(420120, 144)
# X_train은 이미 (샘플 수, timesteps, n_features) 형태이므로 추가 reshape 필요 없음
print("데이터 전처리 완료.")

# --- 2. 모델 구성 및 콜백 설정 ---
print("2. 모델 구성 및 콜백 설정 시작...")
model = Sequential([
    # LSTM 레이어: 100개의 유닛, tanh 활성화 함수, 입력 형태는 (timesteps, n_features)
    LSTM(units=100, activation='tanh', input_shape=(timesteps, n_features), return_sequences=False),
    # Dense 레이어: output_steps(144)개의 값을 한 번에 예측
    Dense(output_steps) 
])

# 모델 컴파일: Adam 옵티마이저, MSE 손실 함수 사용
model.compile(optimizer='adam', loss='mse')

# 모델 저장 경로 설정
path = 'C:/Study25/_save/keras56_jena_climate_2009_2016/'
if not os.path.exists(path): # 폴더가 없으면 생성
    os.makedirs(path)
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 에포크와 검증 손실을 포함한 파일 이름 형식
filepath = os.path.join(path, 'k56_jena_' + filename)

# ModelCheckpoint: 검증 손실이 가장 낮은 모델 가중치만 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=filepath)
# EarlyStopping: 검증 손실이 20 에포크 동안 개선되지 않으면 훈련 조기 중단
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

print("모델 구성 및 콜백 설정 완료.")

# --- 3. 모델 훈련 ---
print("3. 모델 훈련 시작...")
# 모델 훈련: 100 에포크, 배치 사이즈 32, 20%를 검증 데이터로 사용
# callbacks에 EarlyStopping과 ModelCheckpoint 추가
model.fit(X_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2, callbacks=[es, mcp], verbose=1)
print("모델 훈련 완료.")

# --- 4. 예측 수행 ---
print("4. 예측 수행 시작...")

# 예측을 위한 초기 입력 시퀀스 준비:
# '2016-12-31 00:10:00'부터 144개 예측을 위해 필요한 입력 시퀀스 (과거 144개 데이터)
# 이 시퀀스는 '2016-12-30 00:10:00'부터 '2016-12-31 00:00:00'까지의 실제 데이터입니다.
initial_sequence_start_idx = start_idx_original_df - timesteps
initial_sequence_end_idx = start_idx_original_df # 슬라이싱에서 끝 인덱스는 포함되지 않음

# 스케일링된 데이터를 사용하여 예측 입력 시퀀스 생성
# 형태는 (1, timesteps, n_features)여야 합니다.
current_sequence_for_prediction = data_scaled[initial_sequence_start_idx : initial_sequence_end_idx].reshape(1, timesteps, n_features)

# Multi-output 모델은 한 번의 predict 호출로 output_steps (144)개의 값을 예측합니다.
predicted_values_scaled = model.predict(current_sequence_for_prediction, verbose=0)[0] # 첫 번째 (유일한) 샘플의 예측값을 가져옴

# 예측 결과 스케일링 역변환 ('wd (deg)' 스케일러 사용)
predicted_values = wd_scaler.inverse_transform(predicted_values_scaled.reshape(-1, 1))

# 예측 기간의 실제 값 준비 (RMSE 계산용)
actual_start_date = pd.to_datetime('2016-12-31 00:10:00')
actual_end_date = pd.to_datetime('2017-01-01 00:00:00')
# 원본 df에서 해당 기간의 'wd (deg)' 실제 값 추출
actual_data_period = df.loc[actual_start_date:actual_end_date, 'wd (deg)'].values

# 실제 데이터와 예측 데이터의 길이가 다를 경우를 대비 (슬라이싱 오류 방지)
if len(actual_data_period) != len(predicted_values):
    print(f"경고: 실제 데이터({len(actual_data_period)}개)와 예측 데이터({len(predicted_values)}개)의 길이가 다릅니다. 실제 데이터를 예측 길이에 맞춰 자릅니다.")
    actual_data_period = actual_data_period[:len(predicted_values)]

# RMSE (Root Mean Squared Error) 계산
rmse = np.sqrt(mean_squared_error(actual_data_period, predicted_values))
print(f"예측 RMSE (144개): {rmse:.4f}")
print("예측 수행 완료.")

# --- 5. 결과 CSV 파일 저장 ---
print("5. 결과 CSV 파일 저장 시작...")

# 예측된 값 중 처음 100개만 추출하여 제출용으로 사용
submit_predictions = predicted_values[:100]

# 예측값을 DataFrame으로 변환
submit_df = pd.DataFrame(submit_predictions, columns=['wd (deg)'])

# CSV 저장 경로 및 파일명 설정 (***사용자 환경에 맞게 조정 필요***)
submit_path = 'C:/Study25/_data/kaggle/jena/'
submit_filename = 'jena_이상엽_5_submit.csv' # 사용자 이름과 제출 번호에 맞춰 수정 가능
submit_filepath = os.path.join(submit_path, submit_filename)

# 인덱스 없이 CSV 파일로 저장
submit_df.to_csv(submit_filepath, index=False)
print(f"CSV 파일 저장 완료: {submit_filepath}")
print("모든 작업 완료.")
