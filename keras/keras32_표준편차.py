### <<16>>

import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 데이터
data = np.array([[1,  2,  3, 1],
                [4, 5, 6, 2],
                [7, 8, 9, 3],
                [10, 11, 12, 114],
                [13, 14, 15, 115],]
                )
print(data.shape)   # (5,  4)

# 1) 평균
means = np.mean(data, axis=0)
print('평균 :', means)    # 평균 : [ 7.  8.  9. 47.]

# 2) 모집단 분산(n으로 나눈다)
population_variance = np.var(data, axis=0)
print('모집단 분산 : ', population_variance)  # 모집단 분산 :  [  18.   18.   18. 3038.]

# 3) 표본 분산 (n-1로 나눈다)
variences = np.var(data, axis=0, ddof=1)    # ddof : n-1빵 하겠다.
print('표본분산 : ', variences)     # 표본분산 :  [  22.5   22.5   22.5 3797.5]

# 4) 모집단 표준편차
std1 = np.std(data, axis=0)
print('모집단 표준편차 : ', std1)   # 모집단 표준편차 :  [ 4.24264069  4.24264069  4.24264069 55.11805512]

# 5) 표본 표준편차
std2 = np.std(data, axis=0, ddof=1)
print('표본 표준편차 : ', std2)     # 표본 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

# 표본에서는 n-1로 나누는 이유 : 데이터가 충분히 커지면 표본이 모집단보다 작기때문에 작은수를 나눠야한다?

# 6) StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('StandardScaler : \n', scaled_data)
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]