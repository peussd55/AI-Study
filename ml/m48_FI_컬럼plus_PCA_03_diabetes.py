### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import load_diabetes
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
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y,
)

# 2. 모델
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('r2 :', model.score(x_test, y_test))     # r2 : 0.39065385219018145
r2 = model.score(x_test, y_test)
print(model.feature_importances_)
    
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
x_f = pd.DataFrame(x, columns=datasets.feature_names)
x1 = x_f.drop(columns=col_name)
x2 = x_f[col_name]
com_len = len(col_name) # 삭제할 컬럼 갯수

x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
    # stratify=y,
)
print(x1_train.shape, x1_test.shape)    # (353, 7) (89, 7)
print(x2_train.shape, x2_test.shape)    # (353, 3) (89, 3)
print(y_train.shape, y_test.shape)      # (353,) (89,)

print('그냥 한거 r2 :', r2)       # 그냥 한거 r2 : 0.39065385219018145

# PCA로 삭제된 컬럼합치기
for i in range(1, com_len):
    print(f'============ n_component가 {i}개 일때 ================')
    pca = PCA(n_components=i)
    x2_train_pca  = pca.fit_transform(x2_train)
    x2_test_pca = pca.transform(x2_test)
    print(x2_train.shape, x2_test.shape)

    x_train = np.concatenate([x1_train, x2_train_pca], axis=1)
    x_test = np.concatenate([x1_test, x2_test_pca], axis=1)
    print(x_train.shape, x_test.shape)

    model.fit(x_train, y_train)
    
    print('FI_Drop + PCA :', model.score(x_test, y_test))       # (날린거 PCA로 축소하고 합쳤을 때)
    
    # ============ n_component가 1 일때 ================
    # (353, 3) (89, 3)
    # (353, 8) (89, 8)
    # FI_Drop + PCA : 0.4160264085258457
    # ============ n_component가 2 일때 ================
    # (353, 3) (89, 3)
    # (353, 9) (89, 9)
    # FI_Drop + PCA : 0.4197983601993409