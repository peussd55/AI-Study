import pandas as pd
import numpy as np

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]
                     ])

# print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()
data2 = imputer.fit_transform(data)
print(data2)
imputer2 = SimpleImputer(strategy='median')
data3 =imputer2.fit_transform(data)
print(data3)

data1 = pd.DataFrame([[2,np.nan,6,8,10,8],
                     [2,4,np.nan,8,np.nan,4],
                     [2,4,6,8,10,12],
                     [np.nan,4,np.nan,8,np.nan,8],
                     ]).T
data.columns = ['x1','x2','x3','x4']

imputer4 = SimpleImputer(strategy='most_frequent')
data5 =imputer4.fit_transform(data1) #최 빈값(가장 자주 나오는 숫자)
print(data5)


imputer5 = SimpleImputer(strategy='constant', fill_value=777)
data6=imputer5.fit_transform(data) #상수, 특정값
print(data6)

imputer6 = KNNImputer() #knn알고리즘으로 결측치 처리.
data7 = imputer6.fit_transform(data)
print(data7)

####################################################################################################

imputer = IterativeImputer() # 디폴트: bayesianRide 회귀모델
data8 = imputer.fit_transform(data)
print(data8)