### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_breast_cancer
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
print('r2 :', model.score(x_test, y_test))     # r2 : 0.9912280701754386
r2 = model.score(x_test, y_test)
print(model.feature_importances_)
    
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
        # col_name.append(x.columns[i])
    else:
        continue
print(col_name)     # ['mean perimeter', 'mean area', 'mean compactness', 'mean symmetry', 'compactness error', 'concavity error', 'symmetry error', 'fractal dimension error']

# dataframe 생성한 후 삭제할 컬럼 drop하기
x_f = pd.DataFrame(x, columns=datasets.feature_names)
# x_f = x
x1 = x_f.drop(columns=col_name)
x2 = x_f[col_name]
com_len = len(col_name) # 삭제할 컬럼 갯수

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
    stratify=y,
)
print(x1_train.shape, x1_test.shape)    # (1167, 6) (292, 6)
print(x2_train.shape, x2_test.shape)    # (1167, 3) (292, 3)
print(y_train.shape, y_test.shape)      # (1167,) (292,)

print('그냥 한거 r2 :', r2)

# PCA로 삭제된 컬럼합치기
for i in range(1, com_len):
    print(f'============ n_component가 {i} 일때 ================')
    pca = PCA(n_components=i)
    x2_train_pca  = pca.fit_transform(x2_train)
    x2_test_pca = pca.transform(x2_test)
    print(x2_train.shape, x2_test.shape)

    x_train = np.concatenate([x1_train, x2_train_pca], axis=1)
    x_test = np.concatenate([x1_test, x2_test_pca], axis=1)
    print(x_train.shape, x_test.shape)

    model.fit(x_train, y_train)
    
    print('FI_Drop + PCA :', model.score(x_test, y_test))       # (날린거 PCA로 축소하고 합쳤을 때)
    
    # 그냥 한거 r2 : 0.9912280701754386
    # ============ n_component가 1 일때 ================
    # (455, 8) (114, 8)
    # (455, 23) (114, 23)
    # FI_Drop + PCA : 0.9912280701754386
    # ============ n_component가 2 일때 ================
    # (455, 8) (114, 8)
    # (455, 24) (114, 24)
    # FI_Drop + PCA : 0.9912280701754386
    # ============ n_component가 3 일때 ================
    # (455, 8) (114, 8)
    # (455, 25) (114, 25)
    # FI_Drop + PCA : 0.9912280701754386
    # ============ n_component가 4 일때 ================
    # (455, 8) (114, 8)
    # (455, 26) (114, 26)
    # FI_Drop + PCA : 0.9912280701754386
    # ============ n_component가 5 일때 ================
    # (455, 8) (114, 8)
    # (455, 27) (114, 27)
    # FI_Drop + PCA : 0.9912280701754386
    # ============ n_component가 6 일때 ================
    # (455, 8) (114, 8)
    # (455, 28) (114, 28)
    # FI_Drop + PCA : 0.9912280701754386
    # ============ n_component가 7 일때 ================
    # (455, 8) (114, 8)
    # (455, 29) (114, 29)
    # FI_Drop + PCA : 0.9912280701754386