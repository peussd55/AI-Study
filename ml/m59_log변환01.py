### <<39>>

# 주로  y를 스케일 하는 용도 (y가 연속값이 회귀에서 씀). x에서도 MinMax나 Standard 스케일러 대신 사용 가능하다.
# ex) y가 0~78 범위일때 0~4까지 줄어든다.
# 앙상블 모델에서 하나의 범위가 나머지랑 차이가 클경우 하나의 범위를 스케일해준다.
# 로그 변환하는 이유(x, y공통) : 로그변환하면 정규분포처럼 변하게되고, 이상치가 많을경우 오차함수의 값이 크게 나오는것을 방지하는 효과 발생
# 필수조건 : 적용하는 데이터가 연속형(범주형 절대X), 음수가 아닐때

import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=10.0, size=1000)
# 지수분포의 평균(mean) 2.0
# 지수분포의 평균(mean) 10.0

print(data)
print(data.shape)                   # (1000,)
print(np.min(data), np.max(data))   
# 0.0005916304210160359 14.597730516770172  
# 0.003687868464453074 78.99662072557558


log_data = np.log1p(data)   # log 0(무한대)이 나오지 않게 1+x을 취함(log0 -> log1)
# log_data = np.log(data)
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)    # 오른쪽이 긴 분포(실데이터)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5) # 정규분포스러운 분포
plt.title('Log Transformed')

plt.show()

