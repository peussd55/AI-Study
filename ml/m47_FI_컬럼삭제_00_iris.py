### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test))     # acc : 0.9333333333333333
print(model.feature_importances_)
    
# [0.02430454 0.02472077 0.7376847  0.21328996]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.02461671084165573

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
print(col_name)     # ['sepal length (cm)']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_name)

print(x)
#      sepal width (cm)  petal length (cm)  petal width (cm)
# 0                 3.5                1.4               0.2
# 1                 3.0                1.4               0.2
# 2                 3.2                1.3               0.2
# 3                 3.1                1.5               0.2
# 4                 3.6                1.4               0.2
# ..                ...                ...               ...
# 145               3.0                5.2               2.3
# 146               2.5                5.0               1.9
# 147               3.0                5.2               2.0
# 148               3.4                5.4               2.3
# 149               3.0                5.1               1.8

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

model.fit(x_train, y_train)
print('acc2 :', model.score(x_test, y_test))     # acc2 : 0.9333333333333333