### <<14>>

# 22 카피

"""
https://www.kaggle.com/competitions/playground-series-s4e1/overview
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 1.데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.head(10)) # 맨 앞 행 10개만 출력
print(test_csv.head(10)) # 맨 앞 행 10개만 출력

# 결측치 확인
print(train_csv.isna().sum())   # 결측치 없음
print(test_csv.isna().sum())   # 결측치 없음

#  shape 확인
print(train_csv.shape)          # (165034, 13)
print(test_csv.shape)           # (110023, 12)
print(submission_csv.shape)     # (110023, 2)

# 컬럼명 확인
print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
#        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
#        'EstimatedSalary', 'Exited'],

# 문자 데이터 수치화(인코딩)
from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder() # 인스턴스화
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])   # fit 함수 + transform 함친 합친 함수 : 변환해서 적용
# # 아래 2줄이랑 같다.
# le_geo.fit(train_csv['Geography'])                                    # 'Geography' 컬럼을 기준으로 인코딩한다.
# train_csv['Geography'] = le_geo.transform(train_csv['Geography'])     # 적용하고 train_csv['컬럼']에 입력함.
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])

# 테스트 데이터도 수치화해야한다. 위에서 인스턴스가 이미 fit해놨기때문에 transform만 적용한다.
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

print(train_csv.head())

# 변환된 컬럼 데이터의 종류별 갯수 출력(데이터 불균형 확인)
print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'])
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884

train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)
print(test_csv.shape)   # (110023, 10)

x = train_csv.drop(['Exited'], axis=1)  
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    random_state=8282,
    shuffle=True,
)

# ## 컬럼별데이터분포 시각화
# fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
# axes = axes.flatten()

# for i, col in enumerate(x_train.columns):
#     x_train[col].plot(kind='kde', ax=axes[i])
#     axes[i].set_title(f'{col} KDE')

# # 남는 subplot은 제거
# for j in range(i+1, len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()

"""
# 데이터 분할 후 스케일링 적용 (스케일링을 하는 이유는 10.스케일링.txt 참조)
    x를 스케일링하지않고 x_train, x_test 나눠서 x_test는 fit하지 않고 trainsform만 하는 이유 
    스케일링 과정은 fit에서 이루어진다.
    전체 데이터(x)를 동일한 스케일링하게되면 과적합문제가 발생한다.
    test할때  train데이터범위에 포함되는 데이터로만 테스트하면 제대로 된 평가를 할 수없다. 예측이란것은 새로운 데이터도 예상해야하기때문이다.
    test데이터도 train데이터랑 동일한 범위내로 스케일링되면  
    x: 0, 1, 2, 3, 4, 5, 7, 80, 90, 100, 110
    x_train : 0, 1, 2, 3, 4, 7, 80, 100 / 스케일링 범위 : 0~100 -> 여기서 fit하면 0~1값나옴.
    x_test : 5, 90, 110 5, 90 ,110 / 스케일링 범위 : 5~110 -> transform 만하면 0.05, 0.9, 1.1가 나옴(min : 0, max : 100). 여기서 또 fit을 하면 0~1이 되어버린다.
    (1) train으로 학습한 모델을 테스트해야하는데 test데이터를 개별적으로 스케일링(fit)해버리면 데이터스케일링 기준이 달라서 train으로 만든 모델의 성능을 테스트할수없다.
    (2) 모델을 만드는 이유는 학습데이터에 없던 값이 들어오면 예측을하는게 목표이다. 따라서 train에 포함되지않은 데이터(ex : 1.1)에 대한 예측을 하기위해서는 110이 1로 변하는 스케일링이 되어서는 안된다.
    train 데이터 범위가 test 데이터 범위를 포함하면 과적합을 피할 순 없다. (random_sate, shuffle의 필요성)
"""

# (1) 스케일러 일괄적용
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# x_train_scaled = scaler.fit_transform(x_train)      # fit : x_train데이터로 스케일링한다 / transform : 적용한다.
# x_test_scaled = scaler.transform(x_test)            # transform : (이전에 fit한 기준으로)적용한다.
# test_csv_scaled = scaler.transform(test_csv)

# (2) 스케일러 컬럼별로 다르게 적용
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
col_transformer = ColumnTransformer(
    transformers=[
        ('credit', StandardScaler(), ['CreditScore']),
        ('age', StandardScaler(), ['Age']),
        ('balance', RobustScaler(), ['Balance']),
        ('salary', RobustScaler(), ['EstimatedSalary']),
        ('tenure', MinMaxScaler(), ['Tenure']),
        # 아래는 이진/카테고리형: 스케일링 불필요하지만 일괄적용 원하면 MinMaxScaler 사용 가능
        ('geo', MinMaxScaler(), ['Geography']),
        ('gender', MinMaxScaler(), ['Gender']),
        ('numprod', MinMaxScaler(), ['NumOfProducts']),
        ('hascr', MinMaxScaler(), ['HasCrCard']),
        ('active', MinMaxScaler(), ['IsActiveMember']),
    ],
    remainder='passthrough'  # 지정하지 않은 컬럼은 그대로 둠
)
# fit_transform은 학습 데이터에만, transform은 테스트/실제 데이터에만!
x_train_scaled = col_transformer.fit_transform(x_train)
x_test_scaled = col_transformer.transform(x_test)
test_csv_scaled = col_transformer.transform(test_csv)

# # 2.모델구성
# from tensorflow.keras.layers import Dropout
# model = Sequential()
# model.add(Dense(64, input_dim=10, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid')) 

path = './_save/keras28_mcp/08_kaggle_bank/'
model = load_model(path + 'k28_0604_1143_0015-0.3212.hdf5')      # restore_best_weights=True일때 저장한 모델과 가중치 동일
# loss :  0.32381799817085266
# acc :  0.8637272715568542
# accuracy :  0.8637272525297409

#model = load_model(path + 'keras27_3_save_model.h5')


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc', 'AUC'],   # 평가지표 acc, AUC
              ) 
"""
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

path = './_save/keras28_mcp/08_kaggle_bank/'
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

start_time = time.time()
hist = model.fit(x_train_scaled, y_train, 
                 epochs=1000, 
                 batch_size=256,
                 verbose=1, 
                 validation_split=0.3,  # loss는 낮아지는데 val_loss는 올라가는 경우 과적합이므로 여기 수치를 늘려본다.
                 callbacks=[es, mcp],
                 )
end_time = time.time()

print("=============== hist =================")
print(hist)
print("=============== hist.history =================")
print(hist.history) # loss, val_loss, acc, val_acc
print("=============== loss =================")
print(hist.history['loss'])
print("=============== val_loss =================")
print(hist.history['val_loss'])

## 그래프 그리기
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)  # (행, 열, 위치)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('bank Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], c='green', label='acc')
plt.plot(hist.history['val_acc'], c='orange', label='val_acc')
plt.title('bank Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()

plt.tight_layout()  # 간격 자동 조정
#plt.show()  # 윈도우띄워주고 작동을 정지시킨다. 다음단계 계속 수행하려면 뒤로빼던지
"""
# 4. 평가, 예측
results = model.evaluate(x_test_scaled, y_test)
print(results)
print("loss : ", results[0]) 
print("acc : ", results[1])  
y_pred = model.predict(x_test_scaled)

y_pred = np.round(y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy_score) 

#print("걸린시간 :", end_time - start_time, "초") 
"""
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

"""

##### csv 파일 만들기 #####
y_submit = model.predict(test_csv_scaled)
print(y_submit)

print('max:', max(y_submit))
# y_submit = np.round(y_submit)  # 평가지표가 auc일땐 round처리하면 안된다.
# 값이 1인 것들의 개수 출력
count_ones = np.sum(y_submit == 1)
print(f'y_submit에서 1인 값의 개수: {count_ones}')

submission_csv['Exited'] = y_submit
from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path}submission_{current_time}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)

plt.show()