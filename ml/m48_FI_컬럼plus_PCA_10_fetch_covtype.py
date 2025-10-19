### <<37>>

# feature_importances : 트리기반모델에서만 제공

from sklearn.datasets import fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
y = y-1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

# 2. 모델
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("=========", model.__class__.__name__, "========")
print('r2 :', model.score(x_test, y_test))
r2 = model.score(x_test, y_test)
print(model.feature_importances_)
    
print("25%지점 :",np.percentile(model.feature_importances_, 25))   # 25% 지점확인

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
print(col_name)     # 

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
print(x1_train.shape, x1_test.shape)    # 
print(x2_train.shape, x2_test.shape)    # 
print(y_train.shape, y_test.shape)      # 

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
    
    # 그냥 한거 r2 : 0.8686264554271405
    # ============ n_component가 1 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 41) (116203, 41)
    # FI_Drop + PCA : 0.8728346084008158
    # ============ n_component가 2 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 42) (116203, 42)
    # FI_Drop + PCA : 0.8732734955207697
    # ============ n_component가 3 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 43) (116203, 43)
    # FI_Drop + PCA : 0.8733595518187999
    # ============ n_component가 4 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 44) (116203, 44)
    # FI_Drop + PCA : 0.8716040033389844
    # ============ n_component가 5 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 45) (116203, 45)
    # FI_Drop + PCA : 0.871414679483318
    # ============ n_component가 6 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 46) (116203, 46)
    # FI_Drop + PCA : 0.8700635956042443
    # ============ n_component가 7 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 47) (116203, 47)
    # FI_Drop + PCA : 0.8750892834092063
    # ============ n_component가 8 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 48) (116203, 48)
    # FI_Drop + PCA : 0.8728776365498309
    # ============ n_component가 9 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 49) (116203, 49)
    # FI_Drop + PCA : 0.8717761159350447
    # ============ n_component가 10 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 50) (116203, 50)
    # FI_Drop + PCA : 0.8733423405591938
    # ============ n_component가 11 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 51) (116203, 51)
    # FI_Drop + PCA : 0.8756572549762054
    # ============ n_component가 12 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 52) (116203, 52)
    # FI_Drop + PCA : 0.8724817775788921
    # ============ n_component가 13 일때 ================
    # (464809, 14) (116203, 14)
    # (464809, 53) (116203, 53)
    # FI_Drop + PCA : 0.8723526931318468