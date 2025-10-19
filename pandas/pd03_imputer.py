import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.impute import IterativeImputer


data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]
                     ])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN


imputer = SimpleImputer()
data2 = imputer.fit_transform(data)
print(np.round(data2,1))

imputer2 = SimpleImputer(strategy='mean')
data3 = imputer2.fit_transform(data)
print(np.round(data3,1)) # 디폴트가 mean(평균)으로 되어있음 

imputer3 = SimpleImputer(strategy='median')
data4 = imputer3.fit_transform(data)
print(np.round(data4,1)) 

data11 = pd.DataFrame([[2, np.nan, 6, 8, 10, 8],
                     [2, 4, np.nan, 8, np.nan, 4],
                     [2, 4, 6, 8, 10, 12],
                     [np.nan, 4, np.nan, 8, np.nan,8]
                     ]).T

data11.columns = ['x1','x2','x3','x4']

imputer4 = SimpleImputer(strategy='most_frequent') # 범주형에서 자주 사용가능
data5 = imputer4.fit_transform(data11) # 최빈값(가장 자주 나온 값)
print(np.round(data5,1)) 

imputer5 = SimpleImputer(strategy='constant',fill_value=777) # 특정 값을 넣기 위해
data6 = imputer5.fit_transform(data) # 상수 
print(np.round(data6,1)) 


imputer6 = KNNImputer() # KNN 알고리즘으로 결측치 처리.
data7 = imputer6.fit_transform(data)  
print(np.round(data7,1)) 

##################################
# 처음값을 평균값을 넣어서 신뢰도가 낮을수있음.
# 결측치가 있는 컬럼 수만큼  돌아감
imputer = IterativeImputer() # 디폴트 모델 : BayesianRide 회귀 모델
data8 = imputer.fit_transform(data)
print(np.round(data8,1))

