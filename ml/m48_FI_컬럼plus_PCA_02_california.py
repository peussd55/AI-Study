### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

# 2. 모델
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('r2 :', model.score(x_test, y_test))     # r2 : 0.83707103301617
print(model.feature_importances_)
    
# [0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
#  0.0921493  0.10859864]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.044329043477773666

percentile = np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25%이하인놈)을 찾아내기
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else:
        continue
print(col_name)     # ['AveBedrms', 'Population']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x_f = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x_f.drop(columns=col_name)
x2 = x_f[['AveBedrms', 'Population']]
# print(x2)  # [20640 rows x 2 columns]

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
    # stratify=y,
)
print(x1_train.shape, x1_test.shape)    # (16512, 6) (4128, 6)
print(x2_train.shape, x2_test.shape)    # (16512, 2) (4128, 2)
print(y_train.shape, y_test.shape)      # (16512,) (4128,)

# PCA로 삭제된 컬럼합치기
pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
print(x2_train.shape, x2_test.shape)    # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape)      # (16512, 7) (4128, 7)

model.fit(x_train, y_train)
print('FI_Drop + PCA :', model.score(x_test, y_test))   

# r2 : 0.83707103301617                 (그냥한거)
# FI_Drop + PCA : 0.8374852395815212    (날린거 PCA로 축소하고 합쳤을 때)
# r22 : 0.8387939858950988              (그냥 날렸을때) 