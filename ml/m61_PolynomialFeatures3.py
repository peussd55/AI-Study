### <<41>>

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]

pf = PolynomialFeatures(degree=2, 
                        include_bias=False,
                        interaction_only=True,  # 거듭제곱항은 없애고 항끼리의 곱만 반환
                        )  
x_pf = pf.fit_transform(x)
print(x_pf)
# [[ 0.  1.  0.]
#  [ 2.  3.  6.]
#  [ 4.  5. 20.]
#  [ 6.  7. 42.]]

# [[  0.   1.   2.   0.   0.   2.]
#  [  3.   4.   5.  12.  15.  20.]
#  [  6.   7.   8.  42.  48.  56.]
#  [  9.  10.  11.  90.  99. 110.]]
