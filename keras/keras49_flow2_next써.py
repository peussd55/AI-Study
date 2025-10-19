### <<21>>

from tensorflow.keras.datasets import  fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)        # (60000, 28, 28)
print(x_train[0].shape)     # (28, 28)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

augment_size = 100      # 증가시킬 사이즈
aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)  # x_train[0]을 100번 타일붙이듯이 이어붙인다.
print(aaa.shape)        # (100, 28, 28, 1)

datagen = ImageDataGenerator(       # 증폭 준비(아직 실행)
    rescale=1./255,                 # 0 ~ 255 스케일링, 정규화
    horizontal_flip=True,           # 수평 반전 <- 데이터 증폭 또는 변환 / 좌우반전
    # vertical_flip=True,           # 수직 반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,          # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    # height_shift_range=0.1,       # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    rotation_range=15,              # 회전 5도
    # zoom_range=1.2,               # 확대 1.2배
    # shear_range=0.7,              # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    fill_mode='nearest',            # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)   # 다 살리면 쓰레기 이미지까지 생성(증폭)됨.

xy_data = datagen.flow( # 수치화된걸 가져오기
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),    # x데이터
    np.zeros(augment_size),                                                     # y데이터 생성, 전부 정수 0으로 가득찬 y값(원핫인코딩된거아님)
    batch_size=augment_size,
    shuffle=False,
).next()
# .next()를 빼면 ImageDataGenerator 객체 반환 : len이 1
# .next()를 쓰면 객체가 튜플을 반환해서 주므로 x, y 반환 : len이 2

print(xy_data)
print(type(xy_data))            # <class 'tuple'>
print(len(xy_data))             # 2 (이미지 갯수X, 튜플 요소의 갯수임)
# xy_date : ((4차원 넘파이), (1차원 넘파이))
print(xy_data[0].shape)         # (100, 28, 28, 1) : 0번 인덱스(신발)을 100개 만들었으므로 batch가 100이됨
# xy_data[0] : 4차원 넘파이
print(xy_data[1].shape)         # (100,)
# xy_data[1] : 1차원 넘파이
print(xy_data[0][0].shape)      # xy_data[0][0] : 신발모양 증폭된거 100개중에 첫번째꺼 
print(xy_data[0][1].shape)      # xy_data[0][1] : 신발모양 증폭된거 100개중에 두번째꺼 

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')
    plt.axis('off')

plt.show()
