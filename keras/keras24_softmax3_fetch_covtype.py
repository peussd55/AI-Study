### <<12>>

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import ssl
import certifi
# Python이 certifi의 CA 번들을 기본으로 사용하도록 설정
# 이 코드는 Python 3.6 이상에서 잘 작동하는 경향이 있습니다.
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl.SSLContext.set_default_verify_paths = lambda self, cafile=None, capath=None, cadata=None: self.load_verify_locations(cafile=certifi.where())

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

### 추가 전처리
# 스케일링
# StandardScaler 적용 (평균 0, 표준편차 1)
# 연속형 컬럼만 추출
### 추가 전처리
# 연속형 컬럼 인덱스 추출
continuous_cols = [
    'Elevation', 'Aspect', 'Slope', 
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]
continuous_cols_indices = [
    datasets.feature_names.index(col) 
    for col in continuous_cols
]

# 스케일링 적용
scaler = StandardScaler()
x_train[:, continuous_cols_indices] = scaler.fit_transform(
    x_train[:, continuous_cols_indices]
)
x_test[:, continuous_cols_indices] = scaler.transform(
    x_test[:, continuous_cols_indices]
)

# 범주형/희소 컬럼은 스케일링하지 않음
# Wilderness_Area_0 ~ Wilderness_Area_3, Soil_Type_0 ~ Soil_Type_39

# 클래스 가중치
class_counts = [211840, 283301, 35754, 2747, 9493, 17367, 20510]
total = sum(class_counts)
class_weights = {i: total/(count*len(class_counts)) for i, count in enumerate(class_counts)}
###

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
    patience=100,
    restore_best_weights=True,
)

start_time = time.time()
hist = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=256,
    verbose=1,
    validation_split=0.2,
    class_weight=class_weights,
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
y_test= np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

y_pred_f1 = np.argmax(model.predict(x_test), axis=1)
print(y_pred_f1)
f1 = f1_score(y_test, y_pred_f1, average='macro')
print('F1-Score :', f1)

plt.show()