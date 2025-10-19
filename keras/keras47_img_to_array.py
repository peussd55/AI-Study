### <<21>>

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨오기
from tensorflow.keras.preprocessing.image import img_to_array   # 땡겨온 이미지를 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/study25/_data/image/me/'

img = load_img(path + 'me.jpg', target_size=(220, 220), )
print(img)          # <PIL.Image.Image image mode=RGB size=100x100 at 0x20FC5CA35E0>
print(type(img))    # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()        # 내 사진 보기

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (220, 102200, 3) : 3차원
print(type(arr))    # <class 'numpy.ndarray'>

## 3차원 -> 4차원 자원증가 ##
# reshape 방법
# arr = arr.reshape(1, 220, 220, 3)
# print(arr)
# print(arr.shape)    # (1, 220, 220, 3)
# expand_dims 방법
img = np.expand_dims(arr, axis=0)   # 1을 0번째 열에 추가한다.(axis : 열의 인덱스,  1 만 추가 가능)
print(img.shape)      # (1, 220, 220, 3)

## me 폴더에 데이터를 npy로 저장하겠다
np.save(path + 'me.npy', arr=img)