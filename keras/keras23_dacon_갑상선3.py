### <<11>>
"""
https://dacon.io/competitions/official/236488/overview/description
"""
# Conv1D로 포팅
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation, Conv2D, MaxPool2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, BatchNormalization
import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.f1score = self.add_weight(name='f1_score', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positives = tf.cast(true_positives, self.dtype)
        count_true_positives = tf.reduce_sum(true_positives)

        possible_positives = tf.cast(y_true, self.dtype)
        count_possible_positives = tf.reduce_sum(possible_positives)

        predicted_positives = tf.cast(y_pred, self.dtype)
        count_predicted_positives = tf.reduce_sum(predicted_positives)

        precision = count_true_positives / (count_predicted_positives + K.epsilon())
        recall = count_true_positives / (count_possible_positives + K.epsilon())
        f1_cal = 2*(precision*recall)/(precision + recall + K.epsilon())

        self.count.assign_add(1)
        a = 1.0 / self.count
        b = 1.0 - a
        self.f1score.assign(a*f1_cal+b*self.f1score)

    def result(self):
        return self.f1score

    def reset_state(self):
        self.f1score.assign(0)
        self.count.assign(0)

# 1.데이터
path = '../_data/dacon/갑상선/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.head()) # 맨 앞 행 5개만 출력
print(test_csv.head()) # 맨 앞 행 5개만 출력

# 결측치 확인
print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

#  shape 확인
print(train_csv.shape)          # (87159, 15)
print(test_csv.shape)           # (87159, 15)
print(submission_csv.shape)     # (110023, 2)

# 컬럼명 확인
print(train_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer'],

# 타겟 데이터 확인
print(train_csv['Cancer'].value_counts())
# 0    76700
# 1    10459
# Name: Cancer, dtype: int64
# -> 이진분류. 다소의 데이터불균형. 
# -> F1-Score : 이진분류평가지표, 데이터불균형에 강함 (+ 선택사항 : 클래스 가중치)

# 모든 컬럼 value_counts 출력
for col in train_csv.columns[:-1]:  # 마지막 컬럼(Cancer) 제외
    print(f"=== {col} ===")
    print(train_csv[col].value_counts())
    print("-" * 30)

# 문자형 데이터 인코딩
categorical_cols = ['Gender', 'Country', 'Race', 'Family_Background', 
                   'Radiation_History', 'Iodine_Deficiency', 'Smoke', 
                   'Weight_Risk', 'Diabetes']

for col in categorical_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

print(train_csv.head())

# 인코딩된 컬럼별 value_counts 출력
for col in categorical_cols:
    print(f"=== {col} ===")
    print(train_csv[col].value_counts())
    print("-" * 30)
    
# 학습에 필요없는 컬럼 제거
train_csv = train_csv
test_csv = test_csv

# x, y 분리
x = train_csv.drop(['Cancer'], axis=1)  
print(x.shape)  # (87159, 14)
y = train_csv['Cancer']
print(y.shape)  # (87159,)
print(np.unique(y, return_counts=True)) # (array([0, 1], dtype=int64), array([76700, 10459], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.25,
    random_state=517,
    stratify=y,
    shuffle=True,
)

# ## 스케일링1 (x_train = x_train.values 살려야함)
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

## 스케일링2 (x_train = x_train.values 죽여야함)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# MinMaxScaler 적용 컬럼
minmax_cols = ['Age', 'Country', 'Race', 'Nodule_Size', 'TSH_Result', 'T3_Result', 'T4_Result']
# StandardScaler 적용 컬럼 (전체 컬럼에서 minmax_cols 제외)
all_cols = x_train.columns.tolist()
standard_cols = [col for col in all_cols if col not in minmax_cols]
# MinMaxScaler 적용
scaler_minmax = MinMaxScaler()
x_train[minmax_cols] = scaler_minmax.fit_transform(x_train[minmax_cols])
x_test[minmax_cols] = scaler_minmax.transform(x_test[minmax_cols])
test_csv[minmax_cols] = scaler_minmax.transform(test_csv[minmax_cols])
# StandardScaler 적용 (standard_cols가 비어있지 않을 때만)
if standard_cols:
    scaler_standard = StandardScaler()
    x_train[standard_cols] = scaler_standard.fit_transform(x_train[standard_cols])
    x_test[standard_cols] = scaler_standard.transform(x_test[standard_cols])
    test_csv[standard_cols] = scaler_standard.transform(test_csv[standard_cols])

## 클래스 가중치 계산
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# 3차원 reshape
x_train = x_train.values
x_test = x_test.values
x_train = x_train.reshape(-1, 14, 1, 1)  # (87159, 14, 1)
x_test = x_test.reshape(-1, 14, 1, 1)  # (87159, 14, 1)

# # 2. 모델구성 (Conv1D) : 과적합구간 : 335
model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', input_shape=(14, 1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling1D(pool_size=2))

# model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(GlobalMaxPooling1D())

# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.3))

# model.add(Dense(128))
# model.add(BatchNormalization()) 
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model_type = 'Conv1D'
# model.add(Dense(1, activation='sigmoid'))

# 2. 모델구성(Conv2D)
model = Sequential()
# 특징 추출기(Feature Extractor) 확장
model.add(Conv2D(256, (3,3), strides=1, input_shape=(14,1,1), padding='same'))  # [1]
model.add(BatchNormalization())  # 배치 정규화 추가
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 1)))
model.add(Dropout(0.5))  # 드롭아웃 강화

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 1)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())  # Flatten 대체

# 분류기(Classifier) 확장
model.add(Dense(512, activation='relu', kernel_regularizer='l2'))  # L2 정규화 추가
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model_type = 'Conv2D'
model.add(Dense(1, activation='sigmoid')) 


# # 2. 모델구성 (Dense) : 과적합 구간 477
# model = Sequential()
# model.add(Dense(128, input_dim=14))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))  # 추가
# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))  # 추가
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))  # 추가
# model_type = 'Dense'
# model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[F1Score()]
)
es = EarlyStopping( 
    monitor = 'val_f1_score',            
    mode = 'max',               
    patience=300,             
    restore_best_weights=True,  
    verbose=1,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_f1_score',
    factor=0.5,
    patience=100,
    min_lr=1e-7,
    verbose=1
)
import datetime, os
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = '../_save/dacon/갑상선/'
filename = '{epoch:04d}-{val_f1_score:.4f}'
filepath = os.path.join(path, f'k23_{date}_{filename}_{model_type}.hdf5')

print(filepath)

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_f1_score',
    mode = 'max',
    verbose=1,
    save_best_only=True,
    filepath = filepath,    
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=2000, 
                batch_size=64,
                verbose=0, 
                class_weight=class_weight_dict,  # 클래스 가중치 적용
                validation_split=0.2,
                callbacks=[es, reduce_lr, mcp],
                )
end_time = time.time()

## 그래프 그리기
plt.figure(figsize=(9, 6))
plt.plot(hist.history['f1_score'], c='green', label='f1_score')
plt.plot(hist.history['val_f1_score'], c='yellow', label='val_f1_score')
plt.title(f'{model_type}_f1_score')
plt.xlabel('epochs')
plt.ylabel('F1_score')
plt.legend()
plt.grid()

plt.tight_layout()  # 간격 자동 조정

#plt.show()  # 윈도우띄워주고 작동을 정지시킨다. 다음단계 계속 수행하려면 뒤로빼던지

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print("loss : ", results[0]) 
print("f1-score : ", results[1])  

# 검증 데이터로 F1-Score 계산
from sklearn.metrics import f1_score
y_pred = model.predict(x_test)
print(y_pred.shape) # (17432, 1)
y_pred = (y_pred > 0.5).astype(int)
print(y_pred)
f1 = f1_score(y_test, y_pred)

##### csv 파일 만들기 #####
# y_submit = model.predict(test_csv)
test_data_reshaped = test_csv.values.reshape(-1, 14, 1, 1)
y_submit = model.predict(test_data_reshaped)

y_submit = (y_submit > 0.5).astype(int)
submission_csv['Cancer'] = y_submit
from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path}submission_{f1}_{model_type}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)
csv_nm = f'{path}submission_{f1}_{model_type}.csv'
print("걸린 시간 :", round(end_time-start_time, 2)/60, "분")
print("F1-Score : ", f1)
print(csv_nm, '파일을 저장하였습니다')

plt.show()