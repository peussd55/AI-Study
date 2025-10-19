### <<40>>

import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan],    
                    ])
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
from impyute.imputation.cs import mice
data9 = mice(data.values,
             n=10,  # 디폴트 : 5 (노이즈 낀 선형회귀 5번 / n빵)
             seed=777,
             )
print(data9)
# [[ 2.          2.          2.          1.98790929]
#  [ 4.02423084  4.          4.          4.        ]
#  [ 6.          6.00361777  6.          6.00439426]
#  [ 8.          8.          8.          8.        ]
#  [10.         10.00723555 10.         10.02087923]]