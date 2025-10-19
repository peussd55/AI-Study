### <<16>>

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# (N, 5,5,1) 이미지 
# 이미지가 4차원인 이유 : N개의 갯수, 가로 5픽셀, 세로 5픽셀, RGB(1 또는 3) : 갯수+가로해상도+세로해상도+색상(채널) -> 4차원
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5, 5, 3)))     # 아웃풋 10 / 커널사이즈 2X2 / 5,5,1 input shape    =>  (4, 4, 10)
model.add(Conv2D(5, (2,2)))         # (3, 3, 5) : 아웃풋이 3x3해상도이며 5개(필터)를 생성했다
model.add(Conv2D(3, (2,2)))         # (2, 2, 3) : 아웃풋이 2x2해상도이며 3개(필터)를 생성했다
# 인풋레이어에서는 '채널' / 이후 레이어에서는 '필터'라고 부른다

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          130

#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

#  conv2d_2 (Conv2D)           (None, 2, 2, 3)           63

# =================================================================
# Total params: 398
# Trainable params: 398
# Non-trainable params: 0
# _________________________________________________________________
# 첫번째 레이어 Output Shape에서 채널이 10이 나오는 이유? 출력을 10으로 했기때문에 4x4이미지가 10개로 증폭되어 다음 레이어로 전달
# Param : 가중치 + 편향 갯수
# CNN의 weight는 커널사이즈이다
 