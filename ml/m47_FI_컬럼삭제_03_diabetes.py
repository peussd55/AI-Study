### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_diabetes()
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
print('r2 :', model.score(x_test, y_test))     # r2 : 0.39065385219018145
print(model.feature_importances_)
    
# [0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
#  0.03822911 0.10475955 0.3368922  0.07076568]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.0436340281739831

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
print(col_name)     # ['age', 's1', 's3']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_name)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

model.fit(x_train, y_train)
print('r2 :', model.score(x_test, y_test))     # r2 : 0.4517617945199167