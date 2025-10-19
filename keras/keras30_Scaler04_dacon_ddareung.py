### <<14>>

# 29_dacon_ddareung 카피

# https://dacon.io/competitions/open/235576/overview/description 

import numpy as np 
import pandas as pd 
print(np.__version__)   # 1.23.0
print(pd.__version__)   # 2.2.3

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.python.keras.models import Sequential, load_model
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
        x_train = x_train
        x_test = x_test
    else:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

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

    hist = model.fit(x_train, y_train, 
                    epochs=1000, 
                    batch_size=24, 
                    verbose=3, 
                    validation_split=0.1,
                    callbacks=[es],
                    )
           
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
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('r2 스코어 : ', r2)
    print('loss 값 :', loss)
    print('RMSE 값 :', rmse)

    # # submission.csv에 test_csv의 예측값 넣기
    # print(x_train.shape, test_csv.shape)    # (1062, 9) (715, 9)
    # y_submit = model.predict(test_csv)      # strain데이터의 shape와 동일한 컬럼을 확인하고 넣기. (= results)

    # print(y_submit.shape)                   # (715, 1)

    # ############ submission.csv 파일 만들기 // count컬럼값만 넣어주기
    # submission_csv['count'] = y_submit      # submission_csv 의 count컬럼에 y_submit(벡터) 삽입
    # print(submission_csv)

    # from datetime import datetime
    # current_time = datetime.now().strftime('%y%m%d%H%M%S')
    # submission_csv.to_csv(f'{path}submission_{current_time}.csv')
    # #submission_csv.to_csv(path + 'submission_0521_1300.csv')    # 새로운 csv 파일로 생성

"""
    === 현재 적용 스케일러: None ===
    r2 스코어 :  0.6533094488619982
    loss 값 : 2566.422607421875
    RMSE 값 : 50.65987475429519
    
    === 현재 적용 스케일러: MinMax ===
    r2 스코어 :  0.7807456915623106
    loss 값 : 1623.0592041015625
    RMSE 값 : 40.287208592474336

    === 현재 적용 스케일러: Standard ===
    r2 스코어 :  0.7776046207201955
    loss 값 : 1646.311279296875
    RMSE 값 : 40.5747627325626

    === 현재 적용 스케일러: MaxAbs ===
    r2 스코어 :  0.7819620458695136
    loss 값 : 1614.0548095703125
    RMSE 값 : 40.175302757480345

    === 현재 적용 스케일러: Robust ===
    r2 스코어 :  0.7634995535417325
    loss 값 : 1750.72607421875
    RMSE 값 : 41.841677862895295
"""