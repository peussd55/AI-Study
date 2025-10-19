### <<42>>

# 공분산과 평균을 이용해서 데이터를 타원형태의 군집으로그리고 Mahalanobis 거리를 구해서 이상치를 찾는다.
# -> 데이터분포를 타원형태로 모아주고 타원을 벗어나는것은 이상치로 처리
# 통상적으로 IQR은 단항식(컬럼1개)에 유효하지만 다항식(컬럼2개이상)에는 공분산 이용
# 한 컬럼내 이상치가 아닌데 컬럼 2개이상을 한꺼번에 고려하면 있을 수없는 조합의 값일 수 있다.이럴때 공분산을 이용한다.

import numpy as np 
aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12,50])
aaa = aaa.reshape(-1, 1)    # (13, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1] 
# -10, 50 이상치로 표기