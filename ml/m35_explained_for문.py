### <<35>>

# PCA : 주성분분석
# train_test_split => scaling => pca

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, 
    stratify=y, 
)

#### pca가 들어갈때는 보통 pca전에 스케일러함
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# x_train_original = x_train.copy()
# x_test_original = x_test.copy()

for i in range(4): 
   x_train1 = x_train.copy()
   x_test1 = x_test.copy()

   pca = PCA(n_components=i+1)   # n_componets : 몇 개의 컬럼으로 압축할 것인지
   x_train1 = pca.fit_transform(x_train1)
   x_test1 = pca.transform(x_test1)
   # print(x_copiedx)                    
   #   print(x_train.shape)

   # 2. 모델
   model = RandomForestClassifier()

   # 3. 훈련
   model.fit(x_train1, y_train)
   results = model.score(x_test1, y_test)
    #  print(x.shape)
   print(x.shape, '의 score :', results)

    # (150, 4) 의 score : 0.9333333333333333
    # (150, 4) 의 score : 0.8666666666666667
    # (150, 4) 의 score : 0.9333333333333333
    # (150, 4) 의 score : 0.9

evr = pca.explained_variance_ratio_
print('evr :', evr)            
# evr : [0.73584003 0.21914791 0.03986846 0.0051436 ]
# 첫 번째 주성분 (PC1): 전체 분산의 **73.58%**를 설명
# 두 번째 주성분 (PC2): 전체 분산의 **21.91%**를 설명
# 세 번째 주성분 (PC3): 전체 분산의 **3.99%**를 설명
# 네 번째 주성분 (PC4): 전체 분산의 **0.51%**를 설명

print('evr_sum :', sum(evr))    
# evr_sum : 1.0000000000000002
# 모든 주성분을 합하면 100%

evr_cumsum = np.cumsum(evr)
print('누적합 : ', evr_cumsum)
# 누적합 :  [0.73226253 0.95973467 0.99461959 1.        ]

# 시각화
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()