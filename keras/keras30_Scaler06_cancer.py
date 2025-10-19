### <<14>>

# 29_cancer 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

print(type(datasets))   # <class 'sklearn.utils.Bunch'>

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))          # <class 'numpy.ndarray'>

print(x)
"""
[[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
 ...
 [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]
"""
print(y)
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0 ...]
"""

# 0과 1의 갯수가 몇개인지 찾아보기. (데이터가 불균형한지 확인하기위함 -> 불균형할 경우 데이터 증폭 필요 / 평가지표 F1 Score 사용필요)
# 1. 판다스로 찾기
print(pd.value_counts(y))
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
# 1    357
# 0    212
# dtype: int64

# 2. 넘파이로 찾기
# 중복된 값을 제거하고 고유한(unique) 값들만을 모아서 정렬된 NumPy 배열 형태로 반환
print(np.unique(y, return_counts=True))     # (array([0, 1]), array([212, 357], dtype=int64)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.3,
    random_state=8282,
    shuffle=True,
)
print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

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

    # 2.모델구성
    model = Sequential()
    model.add(Dense(64, input_dim=30, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # sigmoid를 써서 y를 0~1 사이값(양수면 0.5이상 / 음수면 0.5이하)으로 한정한 후, 값 출력
    # (중요) 이진분류 모델 출력레이어에는 무조건 sigmoid 사용. 노드 수도 무조건 1개. sigmoid가 중간에 들어가도 된다. 꼭 출력레이어에만 쓰는 활성함수는 아니다.
    # 부동소수점연산을 하기때문에 sigmoid를 쓴다고 못쓸정도로 정보손실이 나는 것은 아니다.

    # 3. 컴파일, 훈련
    # model.compile(loss='mse', optimizer='adam')   # sigmoid의 출력이 0~1사이이기 때문에 거리계산에 제곱함수인 mse를 쓸 수없다.
    model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['acc'],  # 보조지표 : Accuracy (훈련에 영향X). 활성화함수가 sigmoid면 여기서 예측값이 0.5이상이면 1, 0.5미만이면 0 으로 변환후 정확도 계산
                # 또는 metrics=['accuracy']
                ) 
    # (중요) 이진분류모델은 loss함수 무조건 binary_crossentropy 사용
    # crossentropy 의미 : 반만 남기고 반은 날린다.
    # y가 0이나 1에 가까워지면 = y^이 1이나 0에 가까워지면 = 실제정답과 예측이 반대로 갈 수록 = loss가 늘어남 = 손실률이 늘어남


    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping( 
        monitor = 'val_loss',       
        mode = 'min',               
        patience=1000,             
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

    # path = './_save/keras28_mcp/06_cancer/'
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

    start_time = time.time()
    hist = model.fit(x_train, y_train, 
                    epochs=10000, 
                    batch_size=32,
                    verbose=1, 
                    validation_split=0.2,
                    callbacks=[es],
                    )
    end_time = time.time()
    # loss: 0.1017 - acc: 0.9623 - val_loss: 0.4116 - val_acc: 0.9125 -> loss, acc보다 val_loss와 val_acc를 신뢰해야한다. 
    # (종합) 학습단계에서 val_ 지표기반으로 모델을 개량하고, 평가단계에서 r2, mse(회귀), accuracy(분류)로 평가한다.

    # 4. 평가, 예측
    results = model.evaluate(x_test, y_test)    # compile에서 metrics를 추가해서 loss만 나오는게 아니고 Accuracy도 나온다.
    print(results)                          # [0.12165816128253937, 0.9473684430122375] : [loss, Accuracy]
    print("loss : ", round(results[0], 4))  # 소수점 다섯번째자리에서 반올림. loss :  0.14746
    print("acc : ", round(results[1], 4))   # 소수점 다섯번째자리에서 반올림. acc :  0.93567
    # 실제값은 0 또는 1인데 반해, 예측값은 0에서 1사이 실수로 나오는데 acc가 0.9이상 나오는 이유? acc지표가 자체적으로 이진값으로 변환(0.5이상 -> 1, 0.5미만 -> 0)으로 변환 후 연산하기때문.

    y_pred = model.predict(x_test)
    print('y_pred :', y_pred[:10])  #10개만 출력
    # [9.42291319e-02]
    #  ...
    #  [9.37520742e-01]

    from sklearn.metrics import accuracy_score
    accuracy_score = accuracy_score(y_test, np.round(y_pred)) 
    # y_pred = np.round(y_pred) 또는 y_pred = round(y_pred)
    # accuracy_score(y_test, y_pred) : ValueError: Classification metrics can't handle a mix of binary and continuous targets
    # 연속값(y_pred)를 소수점 첫째자리에서 반올림해서 0 또는 1로 변환(y_pred가 0~1사이값이라 반올림하면 0 또는 1만 나옴)
    # 데이터형태가 맞아야한다. accuracy_score는 이진값 정확도를 측정하는 함수이므로 이진값만을 입력받아야한다.
    # 회귀 평가지표(r2_score, mean_squared_errors)가 연속값을 연산하는 것과 다름
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print("acc_score :", accuracy_score)
    #print("걸린시간 :", round(end_time-start_time, 2), '초')

"""
    === 현재 적용 스케일러: None ===
    acc_score : 0.9532163742690059
    
    === 현재 적용 스케일러: MinMax ===
    acc_score : 0.9883040935672515
    
    === 현재 적용 스케일러: Standard ===
    acc_score : 0.9941520467836257
    
    === 현재 적용 스케일러: MaxAbs ===
    acc_score : 0.9766081871345029

    === 현재 적용 스케일러: Robust ===
    acc_score : 0.9707602339181286
"""