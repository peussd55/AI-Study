### <<20>>

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
    horizontal_flip=True,       # 수평 반전 <- 데이터 증폭 또는 변환
    vertical_flip=True,         # 수직 반전 <- 데이터 증폭 또는 변환
    width_shift_range=0.1,      # 평행이동 10% (너비의 10% 이내범위에서 좌우 무작위 이동)
    height_shift_range=0.1,     # 수직이동 10% (높이의 10% 이내범위에서 좌우 무작위 이동)
    rotation_range=5,           # 회전 5도
    zoom_range=1.2,             # 확대 1.2배
    shear_range=0.7,            # 좌표하나를 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(짜부러트리기)
    fill_mode='nearest',        # 물체가 이동해서(잡아당겨져서) 생기게 된 공간을 근처픽셀값에 근접한 값으로 대체
)

test_datagen = ImageDataGenerator(  # 평가데이터는 증폭 또는 변환 하지 않는다. 형식만 맞춰주기위해 정규화한다.
    rescale=1./255,             # 0 ~ 255 스케일링, 정규화
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory( 
    path_train,                 # 작업 경로
    target_size=[200, 200],     # 픽셀크기 일괄조정
    batch_size=10,              # 배치사이즈 : (160, 150, 150, 1) -> (10, 200, 200, 16). default : 32
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
    batch_size=10,              # 배치사이즈 : (120, 150, 150, 1) -> (10, 200, 200, 12)
    class_mode='binary',        # 이진분류
    color_mode='grayscale',     # 흑백(1채널)
    # shuffle=True,             # 필요없음(default : False)
)
# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001887E5312B0>
print(xy_train[0])              # batch_size 만큼 shuffle한 사진(1, 0)를 0번째에 넣어놓은 것을 출력
"""
(array([[[[0.00000000e+00],
         [0.00000000e+00],
         [0.00000000e+00],
         ...,
         [0.00000000e+00],
         [0.00000000e+00],
         [0.00000000e+00]],
         
        [[5.09803966e-02],
         [5.09803966e-02],
         [5.09803966e-02],
         ...,
         [5.49019650e-02],
         [5.49019650e-02],
         [5.40251024e-02]]]], dtype=float32), array([1., 1., 1., 0., 1., 0., 1., 1., 0., 0.], dtype=float32))
         # 첫번째 array : 10개의 증강된 이미지 데이터 묶음(1 80개 + 0 80개 중에 10개)
         # 두번째 array : 정답 레이블. 위 묶음이 요소가 각각 1인지 0인지
         # 0과 1을 섞어서 배치사이즈(10개) 만큼을 담아둔 객체(DirectoryIterator)
""" 

print(len(xy_train))            # 16 (xy_train[i]가 파이썬데이터타입으(tuple)로 파이썬 문법 len 사용해야함)

# (xy_train[i][i] 넘파이데이터타입으로 넘파이 문법 shape)
print(xy_train[0][0].shape)
# xy_train[0]의 첫번째 array
# (10, 200, 200, 1) 
# 텐서형태의 문제지(x)
print(xy_train[0][1].shape)
# xy_train[0]의 두번째 array
# (10,)
# 벡터형태의 정답지(y)

print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'> = 대괄호 대신 소괄호를 쓰는 수정이 안되는 리스트
print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
# print(xy_train.shape)         # AttributeError: 'DirectoryIterator' object has no attribute 'shape'
# print(xy_train[16])           # ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[0][2])         # IndexError: tuple index out of range

"""
- xy_train 이터레이터 객체에 x,y, batch 까지 모두 정의 되어있어서 학습시킬때 이것만 불러와서 학습시키면된다.
- 변환 : 증강을 하기위한 연산 (배치사이즈, 타겟사이즈 등등 정의된 옵션으로 연산)
- 증강 : 연산을 통해 이미지를 생성
- xy_train : 실제 증강을 수행하는 생성기
"""