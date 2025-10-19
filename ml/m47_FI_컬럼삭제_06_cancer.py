### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_breast_cancer
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
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('acc :', model.score(x_test, y_test))     # acc : 0.9912280701754386
print(model.feature_importances_)
    
# [0.01815764 0.01569501 0.         0.0032644  0.00396126 0.00207543
#  0.00426753 0.06830692 0.00221316 0.00328998 0.01382377 0.00926016
#  0.0153579  0.01026187 0.0181306  0.00254898 0.         0.01395431
#  0.00322362 0.00251621 0.00386465 0.02203292 0.6079987  0.05683359
#  0.01485889 0.00435244 0.01172198 0.05729257 0.00671581 0.00401975]
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인
# 25%지점 : 0.0032707941136322916

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
print(col_name)     # ['mean perimeter', 'mean area', 'mean compactness', 'mean symmetry', 'compactness error', 'concavity error', 'symmetry error', 'fractal dimension error']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_name)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

model.fit(x_train, y_train)
print('acc2 :', model.score(x_test, y_test))     # acc2 : 0.9824561403508771