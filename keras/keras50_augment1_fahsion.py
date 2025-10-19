### <<22>>

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

datagen = ImageDataGenerator(       # 증폭 준비(아직 실행)
    # rescale=1./255,                 # 0 ~ 255 스케일링, 정규화
    horizontal_flip=True,           # 수평 반전 <- 데이터 증폭 또는 변환 / 좌우반전
    # vertical_flip=True,           # 수직 반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,          # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    # height_shift_range=0.1,       # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    rotation_range=15,              # 회전 5도
    # zoom_range=1.2,               # 확대 1.2배
    # shear_range=0.7,              # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    # fill_mode='nearest',            # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)   # 다 살리면 쓰레기 이미지까지 생성(증폭)됨.

augment_size = 40000    # 6만개 - > 10만개로 만들 것임

randidx = np.random.randint(x_train.shape[0], size=augment_size)
        # = np.random.randint(60000, 40000)   # 6만개 중에 4만개를 랜덤으로 뽑겠다(특정 사진만 뽑혀서 증폭되는 경우 과적합되기 때문에 랜덤으로 추출)
print(randidx)                              # [33928 44150 11934 ... 48901 39288 43015]
print(np.min(randidx), np.max(randidx))     # 1 59999

# 4만개 데이터생성
x_augmented = x_train[randidx].copy()   # 4만개의 데이터 copy, copy로 새로운 메모리 할당. 서로영향X
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (40000, 28, 28)
print(x_augmented.shape)    # (40000, 28, 28)

# 차원변환
# x_augmented = x_augmented.reshape(40000, 28, 28, 1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],
    1,
    )
print(x_augmented.shape)    # (40000, 28, 28)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0] #튜플에서 x만 추출

print(x_augmented.shape)    # (40000, 28, 28, 1)

# x데이터 reshape
print(x_train.shape)    # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate([x_train, x_augmented])
y_train = np.concatenate([y_train, y_augmented])
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

# y 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (100000, 10) (10000, 10)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1)))    # (28, 28, 1) 에서의 채널(1 또는 3)은 여기서만 쓰이고 다음 레이어부터는 filters를 입력받는다. 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))              
model.add(Conv2D(32, (3,3), activation='relu'))         # activation의 역할 : 레이어의 출력을 한정시킴
model.add(Flatten())                                    # Flatten 하는 이유 : 다중분류를 위해선 softmax를 써야하고 softmax를 쓰기위해선 flatten으로 차원변환을 해야하기때문.
model.add(Dense(units=16, activation='relu'))           # 원하는 평가를 하기위해서 그에 맞는 데이터타입변환 하기위해 Flatten을 사용한다.
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(16,)))
model.add(Dense(units=10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', 
                   mode='min',
                   patience=50,
                   verbose=1,   # stop이 어느 epoch에서 걸렸는지 출력해줌(modelcheckpoint도 적용가능)
                   restore_best_weights=True,
                   )

####################### mcp 세이브 파일명 만들기 #######################
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras50_augment1/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'                # 04d : 정수 4자리, .4f : 소수점 4자리
filepath = "".join([path, 'k50_', date, '_', filename])     # 구분자를 공백("")으로 하겠다.
# ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5
print(filepath)

mcp = ModelCheckpoint(          # 모델+가중치 저장
    monitor = 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = filepath,    # filepath가 고정되지 않았기때문에 val_loss갱신될때 마다 신규파일저장
)
start = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=200, 
                 batch_size=32, 
                 verbose=1, 
                 validation_split=0.2,
                 callbacks=[es, mcp],
                 )
end = time.time()
print("걸린 시간 :", round(end-start, 2), "초") # 496.32 초
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)   # evaluation도 verbose 옵션사용가능
print('loss :', loss[0])
print('acc :', loss[1])

y_pred = model.predict(x_test)
print(y_pred.shape) 
print(y_test.shape) 

# argmax이용하여 최대값있는 인덱스만 벡터(시리즈)로 반환후 얼마나 일치하는지 계산
y_pred = np.argmax(y_pred, axis=1)      # axis = 1 : 행 방향
print(y_pred)           # [1 6 1 ... 1 1 6]
y_test = y_test.values  # nparray로 변환. values : 속성. 값만 반환
y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_pred)
print("accuracy : ", acc) 
# 0.902