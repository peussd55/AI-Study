
"""
https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
"""

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization, Bidirectional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터
# 데이터 load
path = 'c:/Study25/_data/Kaggle/jena/'        
# Date Time을 인덱스로 지정
train_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv')
print(train_csv)

# Date Time을 datetime으로 변환
train_csv['Date Time'] = pd.to_datetime(train_csv['Date Time'], format='%d.%m.%Y %H:%M:%S')

# 파생변수(year, day) 생성 : https://www.tensorflow.org/tutorials/structured_data/time_series?hl=ok 참조
timestamp_s = train_csv['Date Time'].map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

train_csv['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
train_csv['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
train_csv['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
train_csv['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# Date Time을 인덱스로 설정
train_csv.set_index('Date Time', inplace=True)

# 파생변수 4개를 맨 앞으로 이동
first_cols = ['Day sin', 'Day cos', 'Year sin', 'Year cos']
cols = train_csv.columns.tolist()
new_order = first_cols + [col for col in cols if col not in first_cols]
train_csv = train_csv[new_order]

print(train_csv.head())
print(train_csv.shape)      # (420551, 18)
print(train_csv.columns)

"""
Index(['Day sin', 'Day cos', 'Year sin', 'Year cos', 'p (mbar)', 'T (degC)',
       'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
       'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)',
       'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'],
"""

# 결측치 확인 -> 결측치 없음
print(train_csv.info())
print(train_csv.isna().sum()) 
print(train_csv.isnull().sum())

# 컬럼정보출력
# mean : 평균, std : 표준편차, min : 최솟값, 25% : 1사분위, 50% : 2사분위, 75% : 3사분위, max : 최댓값      
print(train_csv.describe().transpose())
"""
                    count         mean        std      min          25%           50%          75%      max
Day sin          420551.0    -0.000078   0.707096    -1.00    -0.707107 -4.544533e-14     0.707107     1.00
Day cos          420551.0    -0.000124   0.707119    -1.00    -0.707107 -7.961906e-14     0.707107     1.00
Year sin         420551.0     0.001614   0.706805    -1.00    -0.704514  3.268522e-03     0.708347     1.00
Year cos         420551.0    -0.000661   0.707408    -1.00    -0.708140 -1.140878e-03     0.707430     1.00
p (mbar)         420551.0   989.212776   8.358481   913.60   984.200000  9.895800e+02   994.720000  1015.35
T (degC)         420551.0     9.450147   8.423365   -23.01     3.360000  9.420000e+00    15.470000    37.28
Tpot (K)         420551.0   283.492743   8.504471   250.60   277.430000  2.834700e+02   289.530000   311.34
Tdew (degC)      420551.0     4.955854   6.730674   -25.01     0.240000  5.220000e+00    10.070000    23.11
rh (%)           420551.0    76.008259  16.476175    12.95    65.210000  7.930000e+01    89.400000   100.00
VPmax (mbar)     420551.0    13.576251   7.739020     0.95     7.780000  1.182000e+01    17.600000    63.77
VPact (mbar)     420551.0     9.533756   4.184164     0.79     6.210000  8.860000e+00    12.350000    28.32
VPdef (mbar)     420551.0     4.042412   4.896851     0.00     0.870000  2.190000e+00     5.300000    46.01
sh (g/kg)        420551.0     6.022408   2.656139     0.50     3.920000  5.590000e+00     7.800000    18.13
H2OC (mmol/mol)  420551.0     9.640223   4.235395     0.80     6.290000  8.960000e+00    12.490000    28.82
rho (g/m**3)     420551.0  1216.062748  39.975208  1059.45  1187.490000  1.213790e+03  1242.770000  1393.54
wv (m/s)         420551.0     1.702224  65.446714 -9999.00     0.990000  1.760000e+00     2.860000    28.49
max. wv (m/s)    420551.0     3.056555  69.016932 -9999.00     1.760000  2.960000e+00     4.740000    23.50
wd (deg)         420551.0   174.743738  86.681693     0.00   124.900000  1.981000e+02   234.100000   360.00
"""

# wv, .max. wv(풍속) 이상치 0으로 치환(가변형 데이터)
wv =  train_csv['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = train_csv['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
print(train_csv.describe().transpose())
"""
                    count         mean        std      min          25%           50%          75%      max
Day sin          420551.0    -0.000078   0.707096    -1.00    -0.707107 -4.544533e-14     0.707107     1.00
Day cos          420551.0    -0.000124   0.707119    -1.00    -0.707107 -7.961906e-14     0.707107     1.00
Year sin         420551.0     0.001614   0.706805    -1.00    -0.704514  3.268522e-03     0.708347     1.00
Year cos         420551.0    -0.000661   0.707408    -1.00    -0.708140 -1.140878e-03     0.707430     1.00
p (mbar)         420551.0   989.212776   8.358481   913.60   984.200000  9.895800e+02   994.720000  1015.35
T (degC)         420551.0     9.450147   8.423365   -23.01     3.360000  9.420000e+00    15.470000    37.28
Tpot (K)         420551.0   283.492743   8.504471   250.60   277.430000  2.834700e+02   289.530000   311.34
Tdew (degC)      420551.0     4.955854   6.730674   -25.01     0.240000  5.220000e+00    10.070000    23.11
rh (%)           420551.0    76.008259  16.476175    12.95    65.210000  7.930000e+01    89.400000   100.00
VPmax (mbar)     420551.0    13.576251   7.739020     0.95     7.780000  1.182000e+01    17.600000    63.77
VPact (mbar)     420551.0     9.533756   4.184164     0.79     6.210000  8.860000e+00    12.350000    28.32
VPdef (mbar)     420551.0     4.042412   4.896851     0.00     0.870000  2.190000e+00     5.300000    46.01
sh (g/kg)        420551.0     6.022408   2.656139     0.50     3.920000  5.590000e+00     7.800000    18.13
H2OC (mmol/mol)  420551.0     9.640223   4.235395     0.80     6.290000  8.960000e+00    12.490000    28.82
rho (g/m**3)     420551.0  1216.062748  39.975208  1059.45  1187.490000  1.213790e+03  1242.770000  1393.54
wv (m/s)         420551.0     2.130191   1.542334     0.00     0.990000  1.760000e+00     2.860000    28.49
max. wv (m/s)    420551.0     3.532074   2.340482     0.00     1.760000  2.960000e+00     4.740000    23.50
wd (deg)         420551.0   174.743738  86.681693     0.00   124.900000  1.981000e+02   234.100000   360.00
"""

# train 데이터 x,y로 분리
selected_features = [
    'Day sin',
    'Day cos',
    'Year sin',
    'Year cos',
    'p (mbar)',        # 기압 → 풍향과 직접적 연관성
    'T (degC)',         # 온도
    'rh (%)',           # 상대습도
    'VPact (mbar)',     # 수증기 압력
    'rho (g/m**3)',     # 공기 밀도
    'wv (m/s)',         # 풍속 → 풍향과의 상호작용 고려
    'max. wv (m/s)',     # 최대 풍속
    'wd (deg)',         # y
]

# train 데이터
xy = train_csv
xy = train_csv[selected_features] # 선택된 특성만 사용
print(xy)
print('x type:',type(xy))    # <class 'pandas.core.frame.DataFrame'>

# nparray로 변환
xy = xy.to_numpy()
print(xy)
print(xy.shape)     # (420551, 12)

# x, y, x_pred, y_true 분리
timesteps = 144
# def split_xy(dataset, timesteps,step):
#     # print('dataset:',dataset)
#     # print('timesteps:',timesteps)
#     aaa = []
#     print('len(dataset):',len(dataset))     # 420407
#     print('timesteps + 1:',timesteps + 1)   # 145
#     for i in range(0, len(dataset) - timesteps + 1, step):   # for i in range(433) : 576 - 144 + 1 = 433
#         # print('len(dataset) - timesteps + 1:', (len(dataset) - timesteps + 1))
#         subset = dataset[i : i+timesteps]
#         # print('i:',i)
#         # print('i+timesteps:',i+timesteps)
#         # print('subset:',subset)
#         aaa.append(subset)
#         # print('aaa:',aaa)
#     return np.array(aaa)

def split_xy(dataset, timesteps, step):
    aaa = []
    total_len = len(dataset)
    print('total_len:', total_len)  # 420407 = 420551 - 144
    # 144스텝으로 나누어떨어지기위해 마지막 288개는 남기고, 그 위 71개는 건너뜀
    exclude_start = total_len - timesteps*2 - 71
    exclude_end = total_len - timesteps
    print('exclude_start:', exclude_start)  # 420048
    print('exclude_end:', exclude_end)      # 420263
    for i in range(0, total_len - timesteps + 1, step):
        # i가 제외구간(즉, [exclude_start, exclude_end))에 해당하면 건너뜀
        if exclude_start <= i < exclude_end:
            continue
        subset = dataset[i : i+timesteps]
        aaa.append(subset)
    return np.array(aaa)


data = split_xy(dataset=xy[:-timesteps], timesteps=timesteps, step=72)
print(data)
print(data.shape)   # (5834, 144, 12)

# x 생성
x = data[:-1, :, :-1]  # x의 마지막 샘플 제거(마지막은 predict으로 들어가야함)하고 마지막 열(y)제거
print(x)
print(x.shape)      # (5833, 144, 11)

# y 생성
y = data[1:, :, -1] # x보다 한스텝만큼 앞에 있는 마지막열
print(y)
print(y.shape)      # (5833, 144)

# x_pred 생성
x_pred = data[-1:, :, :-1]  # x 마지막 샘플(마지막 144 타임스텝)만 추출
print(x_pred)
print(x_pred.shape) # (1, 144, 11)

# y_true(정답지) 생성
y_true = xy[-timesteps:, -1]  # shape: (144,)
print(y_true)
print(y_true.shape) # (144,)
# y_true 2차원으로 변환
y_true = y_true.reshape(1, -1)
print(y_true)
print(y_true.shape) # (1, 144)

# train, test 분할 (10:2)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False  # 시간순서 유지하기 위해 shuffle은 False
)
# train, val 분할 (9:2)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=2/9, shuffle=False  # 시간순서 유지하기 위해 shuffle은 False
)

print("x_train shape:", x_train.shape)  # (3629, 144, 11)
print("x_val shape:", x_val.shape)      # (1037, 144, 11)
print("x_test shape:", x_test.shape)    # (1167, 144, 11)
print("y_train shape:", y_train.shape)  # (3629, 144)
print("y_val shape:", y_val.shape)      # (1037, 144)
print("y_test shape:", y_test.shape)    # (1167, 144)

# # x_train 전체 분포도 확인
# x_train_2d = x_train.reshape(-1, x_train.shape[-1])
# feature_names = [f'feature_{i}' for i in range(x_train.shape[-1])]
# train_df = pd.DataFrame(x_train_2d, columns=feature_names)
# plt.figure(figsize=(14, 7))
# sns.violinplot(data=train_df)
# plt.title('x_train ')
# plt.xticks(rotation=45)
# plt.show()

# x_train 스케일러별 분포도 확인
# x_train_2d = x_train.reshape(-1, x_train.shape[-1])
# # StandardScaler 적용
# scaler = RobustScaler()
# # scaler = RobustScaler()
# x_train_scaled_2d = scaler.fit_transform(x_train_2d)
# # DataFrame으로 변환 (컬럼명은 실제 데이터에 맞게 지정)
# feature_names = [f'feature_{i}' for i in range(x_train.shape[-1])]
# scaled_df = pd.DataFrame(x_train_scaled_2d, columns=feature_names)
# # melt로 long-form 변환
# violin_data = scaled_df.melt(var_name='Feature', value_name='Value')
# # 분포도 시각화
# plt.figure(figsize=(14, 7))
# sns.violinplot(x='Feature', y='Value', data=violin_data)
# plt.title('StandardScaler x_train ')
# plt.xticks(rotation=45)
# plt.show()
# # x_train을 2D로 변환
# x_train_2d = x_train.reshape(-1, x_train.shape[-1])
# # 맨 앞 4개 컬럼 분리
# x_train_first4 = x_train_2d[:, :4]         # 스케일러 제외할 부분
# x_train_rest = x_train_2d[:, 4:]           # 스케일러 적용할 부분
# # 스케일러 적용
# scaler = RobustScaler()
# x_train_rest_scaled = scaler.fit_transform(x_train_rest)
# # 다시 합치기 (맨 앞 4개 + 스케일링된 나머지)
# x_train_scaled_2d = np.hstack([x_train_first4, x_train_rest_scaled])
# # DataFrame 변환
# feature_names = [f'feature_{i}' for i in range(x_train.shape[-1])]
# scaled_df = pd.DataFrame(x_train_scaled_2d, columns=feature_names)
# # melt로 long-form 변환
# violin_data = scaled_df.melt(var_name='Feature', value_name='Value')
# # 분포도 시각화
# plt.figure(figsize=(14, 7))
# sns.violinplot(x='Feature', y='Value', data=violin_data)
# plt.title('RobustScaler x_train (맨 앞 4개 컬럼 제외)')
# plt.xticks(rotation=45)
# plt.show()

# 스케일링
# 1. 3D → 2D 변환 (샘플×타임스텝, 특성)
x_train_2d = x_train.reshape(-1, x_train.shape[-1])
x_val_2d = x_val.reshape(-1, x_val.shape[-1])
x_test_2d = x_test.reshape(-1, x_test.shape[-1])
print(x_train_2d.shape) # (522576, 11)
print(x_val_2d.shape)   # (149328, 11)
print(x_test_2d.shape)  # (168048, 11)

# 2. 스케일러적용
scaler = StandardScaler()
x_train_scaled_2d = scaler.fit_transform(x_train_2d)
x_val_scaled_2d = scaler.transform(x_val_2d)
x_test_scaled_2d = scaler.transform(x_test_2d)

# 3D로 복원 (원래 shape로)
x_train_scaled = x_train_scaled_2d.reshape(x_train.shape)
x_val_scaled = x_val_scaled_2d.reshape(x_val.shape)
x_test_scaled = x_test_scaled_2d.reshape(x_test.shape)

# 3. 2D → 3D 복원
x_train_scaled = x_train_scaled_2d.reshape(x_train.shape) 
x_val_scaled = x_val_scaled_2d.reshape(x_val.shape) 
x_test_scaled = x_test_scaled_2d.reshape(x_test.shape) 
print(x_train_scaled.shape) # (3629, 144, 11)
print(x_val_scaled.shape)   # (1037, 144, 11)
print(x_test_scaled.shape)  # (1167, 144, 11)


# x_train = x_train_scaled
# x_val = x_val_scaled
# x_test = x_test_scaled

print(np.min(x_train), np.max(x_train))     # -23.01 1393.54
print(np.min(x_val), np.max(x_val))         # -10.02 1308.16
print(np.min(x_test), np.max(x_test))       # -13.93 1349.46
# exit()
# 2. 모델구성
model = Sequential()
# model.add(LSTM(64, input_shape=(144, 11), activation='tanh', return_sequences=True))    # return_sequences=True : 3차원(batch, timesteps, feature) 출력 유지
model.add(Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True), input_shape=(144, 11)) )     # Bridirectional 적용. input_shape만 Bidirectonal의 파라미터로 들어가고 나머지는 시계열모델의 파라미터
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear')) 
model.summary()
# RMSE : 54.1 -> 학습속도는 매우 느려졌으나 성능 더 떨어짐

path_mcp = './_save/keras56/'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
# model.load_weights(path_mcp + 'k56_0620_1055_0083.h5')

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# filepath 가변 (갱신때마다 저장)
filename = '{epoch:04d}-{loss:.4f}.h5'    # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path_mcp, 'k56_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# filepath 고정 (종료때만 저장)
# filepath = path + f'keras56_mcp.hdf5'
print(filepath)

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    verbose=1,
    filepath = filepath,
    save_weights_only=True, #가중치만 저장
)

es = EarlyStopping( 
    monitor = 'val_loss',      
    mode = 'min',              
    patience=45,    
    verbose=1,          
    restore_best_weights=True, 
)

date = datetime.datetime.now()
import time
start = time.time()
hist = model.fit(x_train, y_train, 
                epochs=300, 
                batch_size=32, 
                verbose=3, 
                validation_data=(x_val, y_val),
                callbacks=[es,mcp],
                )
end = time.time()

plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')   # plot (x, y, color= ....) : y값만 넣으면 x는 1부터 시작하는 정수 리스트
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
#plt.rcParams['font.family'] = 'Malgun Gothic'
#plt.rc('font', family='Malgun Gothic')
plt.title('jena loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 유측 상단에 라벨표시 
plt.grid()  #격자표시

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 훈련이 끝난 모델의 loss를 한번 계산해서  반환
print('loss : ', loss) 
y_pred = model.predict(x_pred)
print('y_pred :', y_pred)
print(y_pred.shape) # (1, 144, 1)
# y_pred 차원 축소
y_pred = y_pred.squeeze(axis=-1)    # 마지막축(열) 제거
print('y_pred :', y_pred)
print(y_pred.shape)

print('y_true :', y_true)
print(y_true.shape) # (1, 144)

from sklearn.metrics import mean_squared_error
def RMSE(y_true, y_pred):
    # mean_squared_error : mse를 계산해주는 함수
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse = RMSE(y_true, y_pred)
save_filename = f'jena_박은수_submit_{loss}.csv'
print(save_filename, '저장되었습니다.')
print('걸린시간:', end-start, '초')
print('RMSE :', rmse)

### 제출 csv파일 생성 ###
df = pd.read_csv(path + 'jena_climate_2009_2016.csv')
submission_df = df[['Date Time', 'wd (deg)']].tail(144).copy()
submission_df['wd (deg)'] = y_pred.flatten()  # (1,144) → (144,) 변환
submission_df.to_csv(path + f'jena_박은수_submit_{loss}.csv', index=False)

plt.show()
