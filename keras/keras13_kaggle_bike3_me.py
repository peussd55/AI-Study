### <<07>>
"""
https://www.kaggle.com/competitions/bike-sharing-demand
"""

# [Feature Engineering 2]
# 1. train_csv와 new_test.csv로 count 예측
# 이때의 모델 구성은 new_test만들때의 모델과 달라야 한다. 구조가 비슷하기때문에 모델을 비슷하게하면 과적합된다.
# 이전에 모델이 성능이 좋아야 이번 모델링이 의미있다.

import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = './_data/kaggle/bike/'           # 상대경로 : 대소문자 구분X
# path = './/_data//kaggle//bike//'
# path = '.\_data\kaggle\bike\'         # \b가 예약어라서 \b의 \를 슬래쉬로 인식못함. \n \a \b \' 등 예약된 거 빼고 됨.
# path = '.\\_data\\kaggle\\bike\\'

path = 'c:/Study25/_data/kaggle/bike/'  # 절대경로 : 대소문자 구분X

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)


new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)
# casual, registered 컬럼 유무비교 테스트용
#new_test_csv = new_test_csv.drop(['casual', 'registered'], axis=1)

submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

print(train_csv)
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)       # (6493, 10)
print(submission_csv.shape)     # (6493, 2)


# 훈련, 테스트 데이터 컬럼 차이 확인 : datetime컬럼이 공통으로 들어있는 것으로보아 datetime을 인덱스로 하고, test의 count컬럼을 구해야하는 것으로보아 count컬럼이 y가 된다.
print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
print(new_test_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered'],
print(submission_csv.columns)
# Index(['datetime', 'count'],

# 결측치 확인
print(train_csv.info())
print(train_csv.isna().sum())       # 결측치 없음
print(train_csv.isnull().sum())     # 결측치 없음

# 이상치(범위밖 값)를 확인하는데 필요한 기초적인 정보 출력
# mean : 평균, std : 표준편차, min : 최솟값, 25% : 1사분위, 50% : 2사분위, 75% : 3사분위, max : 최댓값      
print(train_csv.describe())    

####### x와 y 분리 ######
x = train_csv.drop(['count'], axis=1)    # y가 될 count는 x에서 제거
# casual, registered 컬럼 유무비교 테스트용
#x = x.drop(['casual', 'registered'], axis=1)    # test셋에는 없는 casual, registered, y제거

print(x)        # [10886 rows x 10 columns] : 데이터프레임(판다스에서의 행렬)
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

# 2. 모델구성
# activation 디폴트는 : 'linear'
model = Sequential()
# casual, registered 컬럼 유무비교 테스트용 : input_dim 10 -> 8
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
epochs = 1000
batch_size = 32
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

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

print('test_size :', test_size)
print('random_state :', random_state)
print('epochs :', epochs)
print('batch_size :', batch_size)
print('loss 값 :', loss)
print('r2 스코어:', r2)
print('rmse 스코어:', rmse)
"""
test_size : 0.2
random_state : 95754
epochs : 300
batch_size : 8
loss 값 : 0.8722174167633057
r2 스코어: 0.9999728288705886
rmse 스코어: 0.9339258040949477
-> 캐글에서 성능 낮아짐 확인 : 1.34 -> 1.36
"""

#### csv 파일 만들기 ####
y_submit = model.predict(new_test_csv)
#print(y_submit)
print(y_submit.shape)      # (6493, 1)
#print(submission_csv)
print(submission_csv.shape) # (6493, 2)

submission_csv['count'] = y_submit
print(submission_csv)

from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path} new_submission_{current_time}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.