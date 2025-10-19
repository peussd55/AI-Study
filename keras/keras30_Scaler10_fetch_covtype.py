### <<14>>

# 29_fetch_covtype 카피

###### scaling (데이터 전처리) ######
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
# import ssl
# import certifi
# # Python이 certifi의 CA 번들을 기본으로 사용하도록 설정
# # 이 코드는 Python 3.6 이상에서 잘 작동하는 경향이 있습니다.
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# ssl.SSLContext.set_default_verify_paths = lambda self, cafile=None, capath=None, cadata=None: self.load_verify_locations(cafile=certifi.where())

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
print(pd.value_counts(y))

y = y.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
print(y.shape) # (581012, 7)

""" 
X데이터 확인
x_df = pd.DataFrame(x, columns=datasets.feature_names)

# 상위 5행 확인
print("\nDataFrame 형태로 확인:")
print(x_df.head())

# 기본 통계 정보(전처리활때 필요한 정보)
print("\n기술 통계:")
pd.set_option('display.max_columns', None)  # 컬럼정보 전부 출력
print(x_df.describe())

# 데이터 타입 및 형태 확인
print("\nDataFrame 구조:")
print(x_df.info())

# 결측치 확인
print("\n결측치 확인:")
print(x_df.isna().sum())    
"""

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.3,
    random_state= 999,
    stratify=y,
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
    model.add(Dense(256, input_dim=54, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    # 3. 컴파일, 훈련
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['acc'],
    )
    
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
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

    # path = './_save/keras28_mcp/10_fetch_covtype/'
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
    hist = model.fit(
        x_train, y_train,
        epochs=200,
        batch_size=512,
        verbose=3,
        validation_split=0.2,
        #class_weight=class_weights,
        callbacks=[es],
    )
    end_time = time.time()
    print('걸린시간 :', end_time-start_time)

    ## 그래프 그리기
    plt.figure(figsize=(18, 5))
    # 첫 번째 그래프
    plt.subplot(1, 2, 1)  # (행, 열, 위치)
    plt.plot(hist.history['loss'], c='red', label='loss')
    plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()

    # 두 번째 그래프
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['acc'], c='green', label='acc')
    plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
    plt.title('acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()  # 간격 자동 조정
    
    # 4. 평가, 예측
    results = model.evaluate(x_test, y_test)
    print('loss:',results[0])
    print('acc:',results[1])

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    y_test_argmax= np.argmax(y_test, axis=1)
    print(y_test_argmax)

    acc = accuracy_score(y_test_argmax, y_pred)
    print('acc :', acc)

    y_pred_f1 = np.argmax(model.predict(x_test), axis=1)
    print(y_pred_f1)
    f1 = f1_score(y_test_argmax, y_pred_f1, average='macro')
    print('F1-Score :', f1)
    
    print(f"\n\n=== 현재 적용 스케일러: {scaler_name} ===")
    print('acc :', acc)
    print('F1-Score :', f1)

    #plt.show()
    
"""
    === 현재 적용 스케일러: None ===
    acc : 0.8713397282908023
    F1-Score : 0.8070136242905617
    
    === 현재 적용 스케일러: MinMax ===
    acc : 0.9156760602166331
    F1-Score : 0.8779952977925313

    === 현재 적용 스케일러: Standard ===
    acc : 0.9251709656691757
    F1-Score : 0.88280182676365

    === 현재 적용 스케일러: MaxAbs ===
    acc : 0.9246029924729209
    F1-Score : 0.888444660424792
    
    === 현재 적용 스케일러: Robust ===
    acc : 0.9277125022948411
    F1-Score : 0.889282225148194
"""