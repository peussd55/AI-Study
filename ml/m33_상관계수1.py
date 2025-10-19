### <<35>>

# 피어슨 상관계수 : 선형데이터에 적용하는 전처리 지표

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y= datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)    # nparray를 판다스 데이터프레임으로 변환
df['target'] = y
print(df)   # [150 rows x 5 columns]

print("================상관관계 히트맵 ==================")
print(df.corr())
"""
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    target
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
target                      0.782561         -0.426658           0.949035          0.956547  1.000000
-> 해석1 : y(target)과 계수(절대값)이 높은 컬럼(x변수)는 중요한 컬럼이므로 놔둔다.
-> 해석2 : x변수간에 계수(절대값)이 높으면 둘 중 하나를 날리는게 좋다.(날려서 성능 좋아지는지는 실제로 확인해봐야함)
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()