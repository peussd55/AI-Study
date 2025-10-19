### <<22>>

# 49-3 카피

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
    np.zeros(augment_size),                                                     # y데이터 생성, 전부 0으로 가득찬 y값
    batch_size=augment_size,
    shuffle=False,
) #.next()
# .next()를 빼면 ImageDataGenerator 객체 반환(반복문에서 호출되면 튜플을 반환) : augment_size/batch_size = 1 이기 때문에 len=1

print(xy_data)                  # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001FA77E34400>
print(type(xy_data))            # <class 'keras.preprocessing.image.NumpyArrayIterator'>
print(len(xy_data))             # 1 : batch_size에 전체증폭데이터(augment_size)를 할당했기때문에 xy_data가 하나만 생성된다.
print(type(xy_data[0]))         # <class 'tuple'> // iterator가 한번 반복해서 (x,y) 튜플반환
# print(type(xy_data[1]))       # error : xy_data가 길이 1이므로 xy_data[1]은 없다.

print(xy_data[0][0].shape)      # (100, 28, 28, 1).  첫번째 신발의 x // xy_data가 길이가 1인 iterator이므로 인덱스 두번 타고 들어가야함.
print(xy_data[0][1].shape)      # (100,). 첫번째 신발의 y // xy_data가 길이가 1인 iterator이므로 인덱스 두번 타고 들어가야함.


plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][0][i], cmap='gray')
    plt.axis('off')

plt.show()
