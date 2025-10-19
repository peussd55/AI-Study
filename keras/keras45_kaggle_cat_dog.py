### <<20>>
# 앵그려봐 submit 까지

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/rules

import numpy as np
import pandas as pd 
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. 데이터
np_path = 'c:/study25/_data/_save_npy/'
path = 'c:/study25/_data/kaggle/cat_dog/'
submission_csv = pd.read_csv(path + 'sample_submission.csv')
# np.save(np_path + 'keras44_01_x_train.npy', arr=x)
# np.save(np_path + 'keras44_01_y_train.npy', arr=y)

# train데이터 로드 (.npy 불러오기)
start = time.time()
# 200하면 터짐
x_train_load = np.load(np_path + "keras44_01_x_train.npy")
y_train_load = np.load(np_path + "keras44_01_y_train.npy")
end = time.time()

# print(x_train)
print(x_train_load[:20])
print(x_train_load.shape, x_train_load.shape)     # (25000, 200, 200, 3) (25000,)
print('train 데이터 불러오기 걸린시간 :', round(end-start, 2), '초')     # 26초

# 로드한 train 데이터를 학습,테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_train_load, y_train_load,
    test_size = 0.3,
    random_state=8282,
)

# # 테스트데이터 로드1 (직접생성)
# # 테스트 이미지 경로 및 데이터 제너레이터 설정
# path_test_dir = 'c:/study25/_data/kaggle/cat_dog/test2/test/'
# test_datagen = ImageDataGenerator(rescale=1./255) # 스케일링만 적용[5]

# # 테스트 파일 목록으로 DataFrame 생성
# test_filenames = os.listdir(path_test_dir)
# test_df = pd.DataFrame({'filename': test_filenames})

# # 제너레이터 생성
# test_generator = test_datagen.flow_from_dataframe(
#     dataframe=test_df,
#     directory=path_test_dir,
#     x_col='filename',
#     y_col=None,
#     target_size=(200, 200),
#     batch_size=2, # 전체 데이터를 한번에 처리
#     class_mode=None,
#     shuffle=False # 순서 유지를 위해 절대 섞지 않음
# )

# submission_x = next(test_generator)

# 테스트데이터 로드2 (.npy 불러오기)
start1 = time.time()
submission_x = np.load(np_path + 'keras44_01_x_submission.npy')
print(f"제출용 테스트 데이터 로드 완료 (from .npy). shape: {submission_x.shape}")
end1 = time.time()
print('test 데이터 불러오기 걸린시간 :', round(end1-start1, 2), '초')      # 16초

# print(submission_x)
print('제출파일 shape:', submission_x.shape)    # (12500, 200, 200, 3)
print('제출파일 type:', type(submission_x))     # <class 'numpy.ndarray'>
print('x_test shape:', x_test.shape)           # (7500, 200, 200, 3)
print('x_test type:', type(x_test))            # <class 'numpy.ndarray'>

# 2. 모델구성 : 레이어 unit이 너무 많으면 메모리 부족으로 학습안된다.
# 학습시 데이터 이동단계 : 학습데이터 cpu메모리에서 gpu메모리로 업로드함 -> gpu메모리(전용메모리)가 부족하면 학습불가 -> 분할배치나 생성자를 사용해야함.
model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, input_shape=(50, 50, 3), activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))       
                                            
model.add(Conv2D(64, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
model.add(MaxPool2D(pool_size=(2, 2))) 
model.add(Dropout(0.25)) 

model.add(Flatten())    
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['acc'],
            )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping( 
    monitor = 'val_loss',       
    mode = 'auto',              
    patience=30,          
    verbose=1,     
    restore_best_weights=True, 
)
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/kaggle/cat_dog/'
# filepath 가변 (갱신때마다 저장)
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k45_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# filepath 고정 (종료때만 저장)
# filepath = path + f'keras46_mcp.hdf5'

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath = filepath,
    verbose=1,
)

start_time = time.time()
# Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5542 MB memory : 할당가능한 gpu memory가 5542 MB
hist = model.fit(x_train, y_train, 
                batch_size = 32,
                epochs=200,
                verbose=3, 
                validation_split=0.2,
                callbacks=[es, mcp],
                )
end_time = time.time()

# 그래프 그리기
plt.figure(figsize=(18, 5))
# 첫 번째 그래프
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('img loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()

# 두 번째 그래프
plt.subplot(1, 2, 2)
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='blue', label='val_acc')
plt.title('img acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.grid()
plt.tight_layout()  # 간격 자동 조정

# 4. 평가, 예측
# 4.1 평가
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])

# loss : 0.44811415672302246
# acc : 0.7915999889373779

# 4.2 예측
from sklearn.metrics import log_loss
y_pred = model.predict(x_test)

local_logloss_score = log_loss(y_test, y_pred)
print(f"\n sklearn.log_loss: {local_logloss_score}")

##### csv 파일 만들기 #####
y_submit = model.predict(submission_x)

submission_csv['label'] = y_submit
# print(submission_csv)

from datetime import datetime
current_time = datetime.now().strftime('%y%m%d%H%M%S')
submission_csv.to_csv(f'{path}submission_{current_time}_{local_logloss_score}.csv', index=False)  # 인덱스 생성옵션 끄면 첫번째 컬럼이 인덱스로 지정됨.(안끄면 인덱스 자동생성)

# # 1. 예측 결과를 DataFrame으로 변환
# # 평가지표가 logloss(binary_crossentropy)이므로 확률 값을 그대로 사용합니다. (round 처리 금지)[1]
# # test_df에 있는 파일명 순서와 y_submit의 예측 결과 순서는 동일합니다.
# submission_df = pd.DataFrame({'id': test_df['filename'], 'label': y_submit.flatten()})

# # 2. 'id' 컬럼을 Kaggle 제출 형식에 맞게 정리
# # '1.jpg' -> '1' 과 같이 파일명에서 숫자만 추출하고 정수형으로 변환합니다.
# submission_df['id'] = submission_df['id'].str.split('.').str[0].astype(int)

# # 3. id 순서대로 정렬 (제출 파일의 표준 형식)
# submission_df = submission_df.sort_values('id')

# # 4. 최종 CSV 파일로 저장
# from datetime import datetime
# current_time = datetime.now().strftime('%y%m%d_%H%M%S')
# # 코드 상단에서 정의한 path 변수를 활용하여 저장
# submission_path = f"{path}submission_catdog_{current_time}.csv"
# submission_df.to_csv(submission_path, index=False)

# print(f"\n제출 파일이 성공적으로 생성되었습니다: {submission_path}")
# print("생성된 파일 미리보기:")
# print(submission_df.head())

plt.show()