### <<41>>

# 52 카피 : Polynomial 추가

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import random, time
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
verbose = 1

# 1. 데이터
path = './_data/kaggle/santander/'           

# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape, test_csv.shape, submission_csv.shape)  # (200000, 202) (200000, 201) (200000, 2)

print(train_csv['target'].value_counts())
# 이진분류 (불균형)
# 0    179902
# 1     20098

# train data x로 y 분리
x = train_csv.drop(['target'], axis=1)
print(x)
y = train_csv['target']

# PolynomialFeatures 적용
# from sklearn.preprocessing import PolynomialFeatures
# pf = PolynomialFeatures(degree=2, include_bias=False)
# x = pf.fit_transform(x)

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y,
)   
# 컬럼이 200개라 PolynomialFeatures 적용하면 메모리 부족으로 오류
# numpy.core._exceptions.MemoryError: Unable to allocate 24.2 GiB for an array with shape (160000, 20300) and data type float64

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
                    tree_method='gpu_hist',
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
start_time = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = verbose,
        #   early_stopping_rounds=10,
          )
end_time = time.time()
print("걸린 시간 : ", end_time-start_time, "초")
print(f"ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("\n")

print("=========", model.__class__.__name__, "========")
print('accuracy_score :', model.score(x_test, y_test))

# [그냥]
# accuracy_score : 

# [PolynomialFeatures]
