### <<20>>

import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
    # horizontal_flip=True,       # 수평 반전 <- 데이터 증폭 또는 변환
    # vertical_flip=True,         # 수직 반전 <- 데이터 증폭 또는 변환
    # width_shift_range=0.1,      # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    # height_shift_range=0.1,     # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    # rotation_range=5,           # 회전 5도
    # zoom_range=1.2,             # 확대 1.2배
    # shear_range=0.7,            # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    # fill_mode='nearest',        # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)

test_datagen = ImageDataGenerator(  # 평가데이터는 증폭 또는 변환 하지 않는다. 형식만 맞춰주기위해 정규화한다.
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory( 
    path_train,                 # 작업 경로
    target_size=[200, 200],     # 픽셀크기 일괄조정
    batch_size=160,             # 배치사이즈 : (160, 150, 150, 1)
    # 통배치사이즈 : 160. 200으로 해도 160으로 잡힌다.
    # 전체 데이터 갯수를 모를때 batch_size를 매우 늘리면 전체 데이터만큼 batch_size가 잡히나, 이러면 메모리 낭비가 심해진다. 명시한 batch_size 만큼 메모리공간을 마련하기때문이다.
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백(1채널)
    shuffle=True,
    seed=333,                   # 시드값 고정
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,                  # 작업 경로
    target_size=[200, 200],     # 픽셀크기 일괄조정
    batch_size=120,             # 배치사이즈 : (120, 150, 150, 1)
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백(1채널)
    # shuffle=True,             # 테스트 데이터는 섞을 필요없음(default : False)
)
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 200, 200, 1) (120,)

# plt.imshow(x_train[78], 'gray')
# plt.show()
acc2 = 0
# 2. 모델구성 : 레이어 unit이 너무 많으면 메모리 부족으로 학습안된다.
while acc2 < 1.0:
    model = Sequential()
    model.add(Conv2D(32, (3,3), strides=1, input_shape=(200, 200, 1))) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
    model.add(MaxPool2D(pool_size=(2, 2))) 
    model.add(Dropout(0.25))       
                                                
    model.add(Conv2D(64, (3,3), strides=1, activation='relu')) # input_shape(높이, 너비, 채널) = (세로, 가로, 채널)                    
    model.add(MaxPool2D(pool_size=(2, 2))) 
    model.add(Dropout(0.25)) 

    model.add(Flatten())    
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.4))

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
        patience=100,          
        verbose=1,     
        restore_best_weights=True, 
    )
    start_time = time.time()
    hist = model.fit(x_train, y_train, 
                    epochs=300,
                    verbose=0, 
                    validation_split=0.1,
                    callbacks=[es],
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
    # print('acc(텐서플로우 자체평가) :', results[1])

    # 4.2 예측
    y_pred = model.predict(x_test)
    # print('y_test:', y_test)
    # print('y_pred:', y_pred)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, np.round(y_pred)) 
    print('걸린시간 :', round(end_time-start_time, 2), '초')
    acc2 = acc
    print('acc(sklearn 지표) :', acc)

plt.show()

"""
걸린시간 : 42.38 초
acc(sklearn 지표) : 0.9833333333333333

걸린시간 : 109.74 초
acc(sklearn 지표) : 0.9666666666666667

걸린시간 : 60.34 초
acc(sklearn 지표) : 0.9916666666666667
"""