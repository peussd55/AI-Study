import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time as time
import datetime

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Dense, Flatten, Dropout ,Conv2D ,Input, MaxPooling2D ,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 데이터 정규화
x_train = x_train/255.
y_train = y_train/255.

#인스턴스화
datagen = ImageDataGenerator(
    #rescale=1./255,        # 0~255 스케일링, 정규화
    horizontal_flip=True,  # 수평 뒤집기  <- 데이터 증폭 또는 변환 /좌우반전
    # vertical_flip=True,    # 수직 뒤집기 <- 데이터 증폭 또는 변환 /상하반전
    width_shift_range=0.1, # 평행이동 10%
    # height_shift_range=0.1,
    rotation_range=15,      
    # zoom_range=1.2,
    # shear_range=0.7,        #좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode='nearest'
) 

#데이터 증폭
augment_size=40000  # 기존 데이터 6만개에서 10만개로 증가

randidx=np.random.randint(x_train.shape[0], size=augment_size)

print(randidx) #[26709 24797 39699 ... 27794 58644 56222]
print(np.min(randidx), np.max(randidx)) #2 59999

x_augmented=x_train[randidx].copy() #4만개의 데이터copy로 새로운 메모리 할당

print(x_augmented.shape) #(40000, 28, 28)
y_augmented = y_train[randidx].copy()

#형태 변경 (40000, 28, 28) -> (40000, 28, 28,1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
# y_augmented = y_augmented.reshape(y_augmented.shape[0],y_augmented.shape[1],y_augmented.shape[2],1)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]


xy_train= train_datagen.flow_from_directory(path_train,               #경로
                                  target_size=(200,200),    #사이즈 규격일치, 큰 사이즈는 축소, 작은사이즈는 확대
                                  batch_size=100,            #
                                  class_mode='binary',
                                  color_mode='rgb',
                                  shuffle=True,
                                  seed=1,
                                  )

#기존데이터 차원 변경
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

#기존 데이터+증강 데이터 합치기
x_train=np.concatenate((x_train, x_augmented))
y_train=np.concatenate((y_train, y_augmented))


; -------------------------------------------------------
np_path='./_data/_save_npy/keras46/gender/'
# np.save(np_path+"keras_44_01_x_trian.npy", arr=x)
# np.save(np_path+"keras_44_01_y_trian.npy", arr=y)

s_time=time.time()
x_train = np.load(np_path+"keras_46_07_x_train_128.npy")
y_train = np.load(np_path+"keras_46_07_y_train_128.npy")

e_time=time.time()

print(x_train.shape, y_train.shape)
print("시간:", round(e_time-s_time,2))

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=121 
)

# print(x_train.shape)# (2647, 250, 250, 3)
# print(x_test.shape)# (662, 250, 250, 3)
# print(y_train.shape)#  (2647,)
# print(y_test.shape)# (662,)
# exit()

path = './_save/keras28_mcp/02_california/'
model = load_model(path + '0604_1145_0336-0.4321.hdf5')

#OnehotEnder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights_dict = dict(enumerate(class_weights))

import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")
filepath = f'./_save/keras39_cifar10{date}' + '.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True, # 가장 좋은 값을 찾아서 저장하기 위한
    filepath=filepath
)

es =  EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=20,
    restore_best_weights=True
)

import os
import glob
import re

def keep_best_model_and_delete_others(folder_path):
    # 폴더 내 .hdf5 파일 리스트
    files = glob.glob(os.path.join(folder_path, '*.hdf5'))

    if not files:
        print("삭제할 체크포인트 파일이 없습니다.")
        return

    # 파일명에서 val_loss 값을 정규표현식으로 추출
    def extract_val_loss(filename):
        basename = os.path.basename(filename)
        match = re.search(r'-(\d+\.\d+)\.hdf5$', basename)
        if match:
            return float(match.group(1))
        else:
            return float('inf')  # val_loss 없으면 무한대로 처리

    # val_loss 기준 최소값 찾기
    best_file = min(files, key=extract_val_loss)
    print(f"✅ 남길 가장 좋은 모델 파일: {best_file}")

    # 가장 좋은 파일 제외 모두 삭제
    for f in files:
        if f != best_file:
            try:
                os.remove(f)
                print(f"삭제 완료: {f}")
            except Exception as e:
                print(f"삭제 실패: {f} - {e}")


epochs=100
batch_size=64

model.fit(x_train, y_train, 
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.2,
          verbose = 1,
          callbacks=[es,mcp],
          class_weight=class_weights_dict
          )

results = model.evaluate(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_labels = (y_pred > 0.5).astype(int).reshape(-1)

print("\n📉 Loss :", results[0])
print("✅ Accuracy :",results[1])

# -------------------- 체크포인트 파일 정리 --------------------
checkpoint_folder = './_save/keras46/'  # 체크포인트 저장 경로
keep_best_model_and_delete_others(checkpoint_folder)


if y_test.ndim == 2 and y_test.shape[1] > 1:
    y_true_labels = np.argmax(y_test, axis=1)
else:
    y_true_labels = y_test.astype(int)
    
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우: 맑은 고딕
num_images = 15

plt.figure(figsize=(15, 8))
for i in range(num_images):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i])
    plt.axis('off')
    
    true_label = y_true_labels[i]
    pred_label = y_pred_labels[i]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"연습: {pred_label}\n 정답: {true_label}", color=color)

plt.suptitle("예측 결과 시각화 (녹색: 정답, 빨강: 오답)", fontsize=16)
plt.tight_layout()
plt.show()


