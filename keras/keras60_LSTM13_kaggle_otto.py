### <<26>>

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization # LSTM 추가
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

le = LabelEncoder()
train_csv['target'] = le.fit_transform(train_csv['target'])

x = train_csv.drop(['target'], axis=1)
y = pd.get_dummies(train_csv['target'])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=8282, shuffle=True, stratify=y
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=5234, stratify=y_train
)

scalers = [
    ('None', None),
    ('MinMax', MinMaxScaler()),
    ('Standard', StandardScaler()),
    ('MaxAbs', MaxAbsScaler()),
    ('Robust', RobustScaler())
]

original_x_train = x_train.copy()
original_x_test = x_test.copy()
original_x_val = x_val.copy()
original_test_csv = test_csv.copy()

for scaler_name, scaler in scalers:
    print(f"\n\n======= 스케일러: {scaler_name} =======")
    x_train = original_x_train.copy().values
    x_test = original_x_test.copy().values
    x_val = original_x_val.copy().values
    test_csv_data = original_test_csv.copy().values

    if scaler is not None:
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)
        test_csv_data = scaler.transform(test_csv_data)

    # 요청사항 1: LSTM 입력을 위한 데이터 Reshape
    # (샘플 수, 93) -> (샘플 수, 93, 1)
    x_train_reshaped = x_train.reshape(x_train.shape[0], 93, 1)
    x_test_reshaped = x_test.reshape(x_test.shape[0], 93, 1)
    x_val_reshaped = x_val.reshape(x_val.shape[0], 93, 1)
    test_csv_reshaped = test_csv_data.reshape(test_csv_data.shape[0], 93, 1)

    # 요청사항 2: 모델의 첫 번째 레이어를 LSTM으로 변경
    model = Sequential()
    model.add(LSTM(128, input_shape=(93, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(9, activation='softmax')) # 클래스 9개

    # 3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
    
    start_time = time.time()
    model.fit(
        x_train_reshaped, y_train,
        epochs=1000, batch_size=64, verbose=0,
        validation_data=(x_val_reshaped, y_val),
        callbacks=[es]
    )
    end_time = time.time()
    
    # 4. 평가, 예측
    print("걸린 시간 :", round(end_time - start_time, 2), "초")
    results = model.evaluate(x_test_reshaped, y_test)
    y_pred = np.argmax(model.predict(x_test_reshaped), axis=1)
    y_test_argmax = np.argmax(y_test.values, axis=1) # y_test가 DataFrame이므로 .values 사용
    acc = accuracy_score(y_test_argmax, y_pred)
    
    print(f"loss: {results[0]}, acc: {results[1]}")
    print(f"Accuracy Score: {acc}")
