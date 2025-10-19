### <<21>>

# 47 카피

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지를 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/study25/_data/image/me/'

img = load_img(path + 'me.jpg', target_size=(200, 200), )
print(img)          # <PIL.Image.Image image mode=RGB size=100x100 at 0x20FC5CA35E0>
print(type(img))    # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()        # 내 사진 보기

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (100, 100, 3) : 3차원
print(type(arr))    # <class 'numpy.ndarray'>

## 3차원 -> 4차원 자원증가 ##
# reshape 방법
# arr = arr.reshape(1, 100, 100, 3)
# print(arr)
# print(arr.shape)    # (1, 100, 100, 3)
# expand_dims 방법
img = np.expand_dims(arr, axis=0)   # 1을 0번째 열에 추가한다.(axis : 열의 인덱스,  1 만 추가 가능)
print(img.shape)      # (1, 100, 100, 3)

# ## me 폴더에 데이터를 npy로 저장하겠다
# np.save(path + 'keras47_me.npy', arr=img)

################### 여기부터 증폭 ###################
datagen = ImageDataGenerator(
    rescale=1./255,               # 0 ~ 255 스케일링, 정규화
    horizontal_flip=True,       # 수평 반전 <- 데이터 증폭 또는 변환 / 좌우반전
    vertical_flip=True,         # 수직 반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,      # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    height_shift_range=0.1,     # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    rotation_range=15,           # 회전 5도
    # zoom_range=1.2,             # 확대 1.2배
    # shear_range=0.7,            # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    fill_mode='nearest',            # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)   # 다 살리면 쓰레기 이미지까지 생성(증폭)됨.

it = datagen.flow(  # img에서 수치가 다 정의되어있어서 여기서 설정할 필요없다
    img,
    batch_size=1,
)

print("===============================================================================")
print(it)           # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000028565AE66D0>
print(type(it))     # <class 'keras.preprocessing.image.NumpyArrayIterator'>
print("===============================================================================")
# aaa = it.next()     # 파이썬 2.0 문법 (파이썬 iter 객체에서는 쓸 수없으나, 케라스 제너레이터 객체에서는 오버라이드되어 사용가능)
# print(aaa)
# print(aaa.shape)      # (1, 200, 200, 3)

# aaa = next(it)
# print(aaa)
# print(aaa.shape)        # (1, 200, 200, 3)

print(it.next())
print(it.next())        # 원래 it 값이 하나면 두번째부터 예외(StopIteration)떠야하나 flow()는 무한 반복(Iteration)이 가능한 이터레이터를 반환
print(it.next())

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5,5))     # fig : Figure 객체 / ax : subplot
for i in range(10):
    # batch = it.next() # IDG에서 랜덤으로 한번 작업 (변환)
    batch = next(it)    # datagen.flow가 반복문만큼 무한 증폭
    print(batch.shape)  # (1, 200, 200, 3)
    batch = batch.reshape(200, 200, 3)
    
    # ax[i].imshow(batch)
    # ax[i].axis('off')     # x,y 축 없애기
    ax[i//5, i%5].imshow(batch)
    ax[i//5, i%5].axis('off')   
    
plt.show()


# fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(5,5))     # fig : Figure 객체 / ax : subplot
# ax = ax.flatten()
# for i in range(10):
#     # batch = it.next() # IDG에서 랜덤으로 한번 작업 (변환)
#     batch = next(it)    # datagen.flow가 반복문만큼 무한 증폭
#     print(batch.shape)  # (1, 200, 200, 3)
#     batch = batch.reshape(200, 200, 3)
    
#     ax[i].imshow(batch)
#     ax[i].axis('off')     # x,y 축 없애기
    
# plt.tight_layout()  
# plt.show()