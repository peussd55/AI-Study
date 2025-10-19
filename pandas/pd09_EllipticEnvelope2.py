### <<42>>

import numpy as np 
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-7000,800,900,1000,210,420,350]]
               ).T
print(aaa.shape)    # (13, 2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.1)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]
# -> EllipticEnvelope는 (10, -7000) 조합과 (50, 350) 조합을 이상치로 간주