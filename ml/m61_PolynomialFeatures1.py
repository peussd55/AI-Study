### <<41>>

# 컬럼증폭 : 다항회귀
# 선형 -> 비선형
# 하는 이유 : 컬럼(특성) 확장 → 선형 모델도 다양한 곡선/비선형 구조를 학습 가능
"""
일반적인 선형 모델은 y = w0 + w1x1 + w2x2 처럼 직선(1차식) 관계만 잘 학습합니다.
현실 데이터는 y = ax² + bx + c 같이 곡선(비선형) 구조를 종종 가집니다.
PolynomialFeatures로 x1², x1x2, x2² 같이 새로운 다항식 컬럼을 생성하면,
모델은 y = w0 + w1x1 + w2x2 + w3x1² + w4x1x2 + w5x2²처럼 각 항마다 가중치를 곱해 합산합니다.
이렇게 되면 선형 모델의 '가중치와 합' 방식이지만, 다양한 곡선 형태(이차, 삼차, 곡선, 꼬인 경향)까지도 표현이 가능합니다.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# 컬럼1 컬럼2

pf = PolynomialFeatures(degree=1, include_bias=False)   
# include_bias : 바이어스 존재 (옵션 디폴트 : True, 값 디폴트 : 1)
# degree : 다항식 차수
# degree=1 : x1, x2
# degree=2 : 1, x1, x2, x1², x1*x2, x2²
# degree=3 : 1, x1, x2, x1², x1*x2, x2², x1³, x1²*x2, x1*x2², x2³

x_pf = pf.fit_transform(x)
print(x_pf)
# degree=2, include_bias=True
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

# degree=2, include_bias=False
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

# degree=3
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]

### 통상적으로
# 선형모델(lr 등)에 쓸 경우에는 include_bias=True를 써서 1만 있는 컬럼을 만드는게 좋음
# 왜냐하면 y = wx + b의  bias = 1의 역할을 하기때문
# 비선형모델(rf, xgb 등)에 쓸 경우에는 include_bias = False가 좋음