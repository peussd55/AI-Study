### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
y = y - 1 # XGB는 0부터 시작하는 클래스 레이블 받음

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test))     # acc : 0.8686264554271405
print(model.feature_importances_)
    
# [0.0954867  0.00713124 0.00411837 0.01321335 0.00743847 0.01333909
#  0.00860075 0.01177858 0.00563742 0.0125805  0.05910649 0.02524424
#  0.03462541 0.02239    0.00331126 0.04878221 0.01951258 0.04187524
#  0.00532929 0.00457224 0.00148059 0.01068641 0.00853365 0.01288974
#  0.0110577  0.04379934 0.01060384 0.00372109 0.00081446 0.00784765
#  0.01112258 0.00627568 0.00522331 0.01550328 0.0191748  0.05527256
#  0.0278828  0.01492529 0.00919063 0.0063742  0.01525284 0.00330649
#  0.02869592 0.0191876  0.02351124 0.04465758 0.0175985  0.0056281
#  0.01753722 0.00303564 0.00975154 0.03427565 0.04038294 0.0107257 ]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.006563458708114922

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
print(col_name)     # ['Slope', 'Hillshade_3pm', 'Soil_Type_0', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_25', 'Soil_Type_27', 'Soil_Type_33', 'Soil_Type_35']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_name)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train, y_train)
print('acc2 :', model.score(x_test, y_test))     # acc2 : 0.8741942979096925