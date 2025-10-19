### <<11>>
"""
https://dacon.io/competitions/official/236488/overview/description
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import backend as K

### f1_score 모니터링 함수
def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (actual_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def custom_f1_score(threshold=0.4):
    """라운딩 제거 + 커스텀 임계값 적용 F1-Score 함수"""
    def f1(y_true, y_pred):
        # 1. 임계값 기반 이진화 (라운딩 제거)
        y_pred_th = K.cast(y_pred > threshold, K.floatx())
        y_true = K.cast(y_true, K.floatx())
        
        # 2. TP/FP/FN 계산
        tp = K.sum(y_true * y_pred_th)
        fp = K.sum(y_pred_th) - tp
        fn = K.sum(y_true) - tp
        
        # 3. Precision & Recall 계산
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        
        # 4. F1-Score 계산
        return 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1
###

# 1.데이터
path = './_data/dacon/갑상선/'

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

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,
    random_state=517,
    shuffle=True,
)

"""
## x_train 컬럼별데이터분포 시각화
# 컬럼 수에 맞게 자동으로 subplot 개수 계산
import math

num_cols = len(x_train.columns)
ncols = 3
nrows = math.ceil(num_cols / ncols)  # 14개 컬럼 → 5행 필요

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
axes = axes.flatten()

for i, col in enumerate(x_train.columns):
    x_train[col].plot(kind='kde', ax=axes[i])
    axes[i].set_title(f'{col}')

# 남는 subplot은 제거
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
"""

## 컬럼별 스케일링
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



### 클래스 불균형 해결을 위한 가중치 계산
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# 모델구성
from tensorflow.keras.layers import Dropout, BatchNormalization
model = Sequential()
model.add(Dense(128, input_dim=14, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))  # 추가
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

## ModelCheckpoint 저장 모델 불러오기
# path = './_save/dacon/갑상선/'
# model = load_model(path + 'keras27_mcp1.hdf5', 
#                      custom_objects={'f1_score': f1_score, 'f1': custom_f1_score()}
#                    )    # compile에 커스텀 metrics 들어가있으면 load할때도 정의해줘야한다.

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=[f1_score, custom_f1_score()],
              ) 

es = EarlyStopping( 
    monitor = 'f1',            
    mode = 'max',               
    patience=30,             
    restore_best_weights=True,  
)
# path_mcp = './_save/dacon/갑상선/'
# mcp = ModelCheckpoint(          # 모델+가중치 저장
#     monitor = 'val_f1',
#     mode = 'max',
#     save_best_only=True,
#     filepath = path_mcp + 'keras23_dacon_갑상선_mcp.hdf5',  # 또는 .h5
# )


start_time = time.time()
hist = model.fit(x_train, y_train, 
                epochs=200, 
                batch_size=32,
                verbose=1, 
                class_weight=class_weight_dict,  # 클래스 가중치 적용
                validation_split=2/9,
                callbacks=[es],
                )
end_time = time.time()

# print("=============== hist =================")
# print(hist)
# print("=============== hist.history =================")
# print(hist.history) # loss, val_loss, acc, val_acc
# print("=============== loss =================")
# print(hist.history['loss'])
# print("=============== val_loss =================")
# print(hist.history['val_loss'])

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

# # 두 번째 그래프
# plt.subplot(1, 2, 2)
# plt.plot(hist.history['acc'], c='green', label='acc')
# plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
# plt.title('acc')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.grid()

# 세번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['f1_score'], c='green', label='f1_score')
plt.plot(hist.history['val_f1_score'], c='yellow', label='val_f1_score')
plt.title('f1_score')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()

plt.tight_layout()  # 간격 자동 조정

#plt.show()  # 윈도우띄워주고 작동을 정지시킨다. 다음단계 계속 수행하려면 뒤로빼던지

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print("loss : ", results[0]) 
print("f1_score : ", results[1])  

# 검증 데이터로 F1-Score 계산
from sklearn.metrics import f1_score
y_pred = model.predict(x_test)
print(y_pred.shape) # (17432, 1)
y_pred = (y_pred > 0.4).astype(int)
print(y_pred)
f1 = f1_score(y_test, y_pred)
print("F1-Score : ", f1)

##### csv 파일 만들기 #####
y_submit = model.predict(test_csv)
y_submit = (y_submit > 0.5).astype(int)
submission_csv['Cancer'] = y_submit
from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path}submission_{current_time}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)

# 모델 저장
path_h5 = './_save/dacon/'
model.save(path_h5 + '갑상선/'+ str(f1) + f'_{current_time}.h5')  # 학습가중치 저장

plt.show()