### <<08>>
"""
https://www.kaggle.com/competitions/bike-sharing-demand
"""

# [Feature Engineering 1]
# 1. train_csv에서 casual과 resistered를 y로 잡는다.
# 2. 훈련해서, test_csv의 casual과 registered를 예측(predict)한다.
# 3. 예측한 casual과 registered를 test_csv에 컬럼으로 넣는다.
#   (N, 8) -> (N, 10) test.csv파일을 new_test.csv파일을 만든다.

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
#y = train_csv['count']
y = train_csv[['count', 'registered']]
print(y)
print(y.shape)  # (10886,2)

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
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='relu'))
model.add(Dense(2, activation='relu'))

# 3. 컴파일, 훈련
epochs = 300
batch_size = 16
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
random_state : 999
epochs : 300
batch_size : 16
loss 값 : 19539.669921875
r2 스코어: 0.2623752211100027
rmse 스코어: 139.78436346067025
"""

#### csv 파일 만들기 ####
y_submit = model.predict(test_csv)
#print(y_submit)
print(y_submit.shape)      # (6493, 2)
print('test_csv 타입 :', type(test_csv))    # <class 'pandas.core.frame.DataFrame'>
print('y_submit 타입 :', type(y_submit))    # <class 'numpy.ndarray'>

newt_test_csv = test_csv     # 메모리 공유 문제때문에 원래는 copy() 사용해야한다.
print('newt_test_csv :')
print(newt_test_csv)

#submission_csv['count'] = y_submit
newt_test_csv[['casual', 'registered']] = y_submit  # 판다스의 데이터타입은 넘파이로 이루어져있기때문에 판다스에 넘파이를 대입할 수있다.
print('newt_test_csv :')
print(newt_test_csv)
# 헤더 = 컬럼

from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
newt_test_csv.to_csv(f'{path}new_test.csv', index=True)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.