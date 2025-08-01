### <<16>>

# 30-5 카피

"""
https://www.kaggle.com/competitions/bike-sharing-demand
"""

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = './_data/kaggle/bike/'           # 상대경로 : 대소문자 구분X
# path = './/_data//kaggle//bike//'
# path = '.\_data\kaggle\bike\'         # \b가 예약어라서 \b의 \를 슬래쉬로 인식못함. \n \a \b \' 등 예약된 거 빼고 됨.
# path = '.\\_data\\kaggle\\bike\\'

path = 'c:/Study25/_data/kaggle/bike/'  # 절대경로 : 대소문자 구분X

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print(train_csv)
print(train_csv.shape)          # (10886, 11)
print(test_csv.shape)           # (6493, 8)
print(submission_csv.shape)     # (6493, 2)

# 훈련, 테스트 데이터 컬럼 차이 확인 : datetime컬럼이 공통으로 들어있는 것으로보아 datetime을 인덱스로 하고, test의 count컬럼을 구해야하는 것으로보아 count컬럼이 y가 된다.
print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
print(test_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed'],
print(submission_csv.columns)
# Index(['datetime', 'count'],
# 차이나는 컬럼 : casual, registered -> 지금은 삭제로 전처리한다.

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())       # 결측치 없음
print(train_csv.isnull().sum())     # 결측치 없음

# 이상치(범위밖 값)를 확인하는데 필요한 기초적인 정보 출력
# mean : 평균, std : 표준편차, min : 최솟값, 25% : 1사분위, 50% : 2사분위, 75% : 3사분위, max : 최댓값      
print(train_csv.describe())    

####### x와 y 분리 ######
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)    # test셋에는 없는 casual, registered, y가 될 count는 x에서 제거
print(x)        # [10886 rows x 8 columns] : 데이터프레임(판다스에서의 행렬)
y = train_csv['count']
print(y)
print(y.shape)  # (10886,) : 시리즈(판다스에서의 벡터)

# 1. 데이터
test_size = 0.2
random_state = 999
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=test_size,
    random_state=random_state,
)

print('x_train.shape: ', x_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

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
    # activation 디폴트는 : 'linear'
    model = Sequential()
    model.add(Dense(128, input_dim=8))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu')) 

    # 3. 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam')

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping( 
        monitor = 'val_loss',       
        mode = 'min',               
        patience=100,             
        restore_best_weights=True,  
    )
    # ####################### mcp 세이브 파일명 만들기 #######################
    # import datetime
    # date = datetime.datetime.now()
    # print(date)         # 2025-06-02 13:00:40.661379
    # print(type(date))   # <class 'datetime.datetime'>
    # date = date.strftime('%m%d_%H%M')
    # print(date)         # 0602_1305
    # print(type(date))   # <class 'str'>

    # path = './_save/keras28_mcp/05_kaggle_bike/'
    # filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
    # filepath = "".join([path, 'k28_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
    # # ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5

    # print(filepath)

    # #exit()

    # mcp = ModelCheckpoint(          # 모델+가중치 저장
    #     monitor = 'val_loss',
    #     mode = 'auto',
    #     save_best_only=True,
    #     filepath = filepath,
    # )

    epochs = 1000
    batch_size = 24
    verbose = 3
    validation_split = 0.1
    hist = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    verbose=verbose, 
                    validation_split=validation_split,
                    callbacks=[es],
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
    plt.title('kaggle Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')   # 유측 상단에 라벨표시 
    plt.grid()  #격자표시
    #plt.show()


    # 4. 평가, 예측
    loss = model.evaluate(x_test, y_test)
    results = model.predict(x_test)     # 예측된 결과(y) 반환
    print('results :', results)

    # 테스트용 y와 예측된 y가 얼마나 유사한지 비교
    # r2_score
    r2 = r2_score(y_test, results)
    # rmse
    def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))
    rmse = RMSE(y_test, results)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('loss 값 :', loss)
    print('r2 스코어:', r2)
    print('rmse 스코어:', rmse)

    # #### csv 파일 만들기 ####
    # y_submit = model.predict(test_csv)
    # #print(y_submit)
    # print(y_submit.shape)      # (6493, 1)
    # #print(submission_csv)
    # print(submission_csv.shape) # (6493, 2)

    # submission_csv['count'] = y_submit
    # print(submission_csv)

    # from datetime import datetime
    # current_time = datetime.now().strftime('%y%m%d%H%M%S')
    # submission_csv.to_csv(f'{path}submission_{current_time}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)

"""
    (1) dropout X
    === 현재 적용 스케일러: None ===
    loss 값 : 21983.400390625
    r2 스코어: 0.30622246037599876
    rmse 스코어: 148.26799308451893
    
    === 현재 적용 스케일러: MinMax ===
    loss 값 : 21949.5703125
    r2 스코어: 0.3072900226651052
    rmse 스코어: 148.1538741868738
    
    === 현재 적용 스케일러: Standard ===
    loss 값 : 21110.892578125
    r2 스코어: 0.333757782636656
    rmse 스코어: 145.29590207174107
    
    === 현재 적용 스케일러: MaxAbs ===
    loss 값 : 21257.2578125
    r2 스코어: 0.3291387290124047
    rmse 스코어: 145.79869994777033

    === 현재 적용 스케일러: Robust ===
    loss 값 : 21602.564453125
    r2 스코어: 0.31824122996297133
    rmse 스코어: 146.97810979088973
    ================================================
    (2) dropout 적용
    === 현재 적용 스케일러: None ===
    loss 값 : 23959.318359375
    r2 스코어: 0.24386400917114903
    rmse 스코어: 154.78798485128718
    
    === 현재 적용 스케일러: MinMax ===
    loss 값 : 21906.037109375
    r2 스코어: 0.30866375799133816
    rmse 스코어: 148.00689693710672
    
    === 현재 적용 스케일러: Standard ===
    loss 값 : 22923.328125
    r2 스코어: 0.2765590654813729
    rmse 스코어: 151.40451647705643
    
    === 현재 적용 스케일러: MaxAbs ===
    loss 값 : 22983.373046875
    r2 스코어: 0.274664109512591
    rmse 스코어: 151.60267865381948    
    
    === 현재 적용 스케일러: Robust ===
    loss 값 : 22940.962890625
    r2 스코어: 0.2760026024158302
    rmse 스코어: 151.462734655167
    => dropout 성능 비슷
"""