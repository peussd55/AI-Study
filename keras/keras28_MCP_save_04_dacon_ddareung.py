### <<14>>

# 19-4 카피

# https://dacon.io/competitions/open/235576/overview/description 

import numpy as np 
import pandas as pd 
print(np.__version__)   # 1.23.0
print(pd.__version__)   # 2.2.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정 -> 데이터프레임 컬럼에서 제거하고 인덱스로 지정해줌.
print(train_csv)        # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
# test_csv는 predict의 input으로 사용한다.
print(test_csv)         # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)   # [715 rows x 1 columns]

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)       # (715, 9)
print(submission_csv.shape) # (715, 1)
# train_csv : 학습데이터
# test_csv : 테스트데이터
# submission_csv : test_csv를 predict하여 예측한 값을 넣어서 제출 

print(train_csv.columns) 
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())     # non-null수 확인(rows와 비교해서 결측치 수 확인), 데이터 타입 확인

print(train_csv.describe()) # 컬럼별 각종 정보확인할 수 있음 (평균,최댓값, 최솟값 등)

# 1. 데이터

######################################## 결측치 처리 1. 삭제 ########################################
# print(train_csv.isnull().sum())       # 컬럼별 결측치의 갯수 출력
print(train_csv.isna().sum())           # 컬럼별 결측치의 갯수 출력

# train_csv = train_csv.dropna()        # 결측치 제거
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)                      # [1328 rows x 10 columns]

######################################## 결측치 처리 2. 평균값 넣기 ########################################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())

########################################  test_csv 결측치 확인 및 처리 ########################################
# test_csv는 결측치 있을 경우 절대 삭제하면 안된다. 답안지에 해당하는(submission_csv)에 채워넣으려면 갯수가 맞아야한다.
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print('test_csv 정보:', test_csv)
print('ㅡㅡㅡㅡㅡㅡ')

x = train_csv.drop(['count'], axis=1)   # axis = 1 : 컬럼 // axis = 0 : 행
print(x)    # [1459 rows x 9 columns] : count 컬럼을 제거

y = train_csv['count']      # count 컬럼만 추출
print(y.shape)  # (1469,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2, 
    random_state=999
)

print('x_train.shape: ', x_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=9))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear')) 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'min',               
    patience=100,             
    restore_best_weights=True,  
)
####################### mcp 세이브 파일명 만들기 #######################
import datetime
date = datetime.datetime.now()
print(date)         # 2025-06-02 13:00:40.661379
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date)         # 0602_1305
print(type(date))   # <class 'str'>

path = './_save/keras28_mcp/04_dacon_ddareung/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k28_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5

print(filepath)

#exit()

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
)

hist = model.fit(x_train, y_train, 
                 epochs=1000, 
                 batch_size=24, 
                 verbose=3, 
                 validation_split=0.1,
                 callbacks=[es,mcp],
                 )
print("=============== hist =================")
print(hist)     # <keras.callbacks.History object at 0x00000179B5A08BB0>
print("=============== hist.history =================")
print(hist.history)
# {} : 딕셔너리 // [] : 리스트
print("=============== loss =================")
print(hist.history['loss'])
print("=============== val_loss =================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')   # plot (x, y, color= ....) : y값만 넣으면 x는 1부터 시작하는 정수 리스트
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('dacon Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 유측 상단에 라벨표시 
plt.grid()  #격자표시
plt.show()


# 목표 : r > 0.58 / loss < 2400

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
#print('result 값 :', results)

r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):
    # mean_squared_error : mse를 계산해주는 함수
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, results)

print('r2 스코어 : ', r2)
print('loss 값 :', loss)
print('RMSE 값 :', rmse)

# submission.csv에 test_csv의 예측값 넣기
print(x_train.shape, test_csv.shape)    # (1062, 9) (715, 9)
y_submit = model.predict(test_csv)      # strain데이터의 shape와 동일한 컬럼을 확인하고 넣기. (= results)

print(y_submit.shape)                   # (715, 1)

############ submission.csv 파일 만들기 // count컬럼값만 넣어주기
submission_csv['count'] = y_submit      # submission_csv 의 count컬럼에 y_submit(벡터) 삽입
print(submission_csv)

from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path}submission_{current_time}.csv')
#submission_csv.to_csv(path + 'submission_0521_1300.csv')    # 새로운 csv 파일로 생성

"""
EarlyStopping X
epochs=1000, 
batch_size=24, 
loss 값 : 1880.01708984375
r2 스코어 :  0.7460340393297711
RMSE 값 : 43.359160485668475

EarlyStopping O / restore_best_weights=True
patience=100
epochs=1000, 
batch_size=24, 
stop지점 : 273,
loss 값 : 2568.28759765625
r2 스코어 :  0.6530575633622454
RMSE 값 : 50.67827469468827

EarlyStopping O / restore_best_weights=False
patience=100
epochs=1000, 
batch_size=24, 
stop지점 : 354,
loss 값 : 2669.23681640625
r2 스코어 :  0.6394206170649933
RMSE 값 : 51.66465749941344
"""

