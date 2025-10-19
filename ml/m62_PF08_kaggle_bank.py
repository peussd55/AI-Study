### <<41>>

# 52 카피 : Polynomial 추가

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

# pandas 컬럼명 불일치경고 무시 (x가 pd.dataframe일때 사용)
import warnings
warnings.filterwarnings('ignore', message='X has feature names, but SelectFromModel was fitted without feature names')

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'logloss'
verbose = 0

# 1. 데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# 인코딩
le_geo = LabelEncoder()
le_gen = LabelEncoder()
train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography']) +1
train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender']) +1
test_csv['Geography'] = le_geo.transform(test_csv['Geography']) +1
test_csv['Gender'] = le_gen.transform(test_csv['Gender']) +1

# 불필요 컬럼 제거
train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId", "Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

print(x.shape, y.shape)     # (165034, 10) (165034,)
print(np.unique(y, return_counts=True))     # (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))

# PolynomialFeatures 적용
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)

# x = pf.fit_transform(x)

# PolynomialFeatures 적용(수치형에만 적용)
x_num = x.drop(['Gender', 'Geography'], axis=1).values
x_cat = x.values
# 수치형 컬럼에만 다항식 변환
x_num_poly = pf.fit_transform(x_num)
x = np.concatenate([x_num_poly, x_cat], axis=1)

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)

# 2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = metric_name,  # 회귀 : rmse, rme, rmsle
                                # 다중분류 : mloglos, merror
                                # 이진분류 : logloss, error

    data_name = 'validation_0', # fit에서 eval_set 몇번째 인덱스로 검증할건지 옵션
    # save_best = True,         # 2.x 버전에서 deprecated
)

model = XGBClassifier(
                    n_estimators = 500,
                    max_depth = 6,
                    gamma = 0,
                    min_child_weight = 0,
                    subsample = 0.4,
                    reg_alpha = 0,
                    reg_lambda = 1,
                    random_state=seed,                      
                    
                    eval_metric = metric_name,  # 회귀 : rmse, rme, rmsle
                                                # 다중분류 : mloglos, merror
                                                # 이진분류 : logloss, error
                                                # 2.1.1버전 이후로 사용하는 위치가 fit에서 model로 위치이동
                    
                    callbacks = [es],
                    )
print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = verbose,
          )
print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("\n")

print("=========", model.__class__.__name__, "========")
print('accuracy_score :', model.score(x_test, y_test))

# [그냥]
# accuracy_score : 0.8624231223679826



# [PolynomialFeatures] degree 2
# accuracy_score : 0.8613627412367074

# [PolynomialFeatures] degree 3
# accuracy_score : 0.8610597751992002

# [PolynomialFeatures] degree 4
# accuracy_score : 0.8608476989729451



# [PolynomialFeatures] : degree2 / label +1
# accuracy_score : 0.8612415548217045

# [PolynomialFeatures] : degree3 / label +1
# accuracy_score : 0.8626654951979883

# [PolynomialFeatures] : degree4 / label +1
# accuracy_score : 0.8613930378404581


# [PolynomialFeatures] : degree2 / 수치형에만 적용
# 0.8630896476504983

# [PolynomialFeatures] : degree2 / 수치형에만 적용 / label + 1
# 0.8630896476504983