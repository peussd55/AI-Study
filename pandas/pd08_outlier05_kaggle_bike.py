### <<42>>

# 60 카피

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
from sklearn.preprocessing import PolynomialFeatures

# pandas 컬럼명 불일치경고 무시 (x가 pd.dataframe일때 사용)
import warnings
warnings.filterwarnings('ignore', message='X has feature names, but SelectFromModel was fitted without feature names')

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'rmse'
verbose = 0

# 1. 데이터
path = './_data/kaggle/bike/'
# 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
print(x)
y = train_csv['count']
print(y)
print(y.shape)
x = x.values

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = RobustScaler()
# x = scaler.fit_transform(x)

# 1.1 이상치 관측처리
def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25,50,75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    outlier_idx = np.where((data > upper_bound) | (data < lower_bound))
    return outlier_idx, iqr, lower_bound, upper_bound

# 컬럼 전체에 대해 자동 반복
for col in range(x.shape[1]):
    col_data = x[:, col]
    outlier_loc, iqr, low, up = outlier(col_data)
    # print(f"컬럼 {col} ({feature_names[col]})")
    print(f"  1사분위: {np.percentile(col_data, 25):.3f},  3사분위: {np.percentile(col_data, 75):.3f}")
    print(f"  IQR: {iqr:.3f}, 하한: {low:.3f}, 상한: {up:.3f}")
    print(f"  이상치 인덱스: {outlier_loc[0]}\n")

# 이상치 중위값처리
def replace_outliers_with_median(x):
    """
    x : 2차원 numpy array (shape: [n_samples, n_features])
    각 열별로 이상치(1.5*IQR 밖의 값)를 해당 열의 median으로 대체
    반환값: 이상치 처리 후 array (원본은 복사됨)
    """
    x_new = x.copy()
    for col in range(x.shape[1]):
        col_data = x[:, col]
        q1, q2, q3 = np.percentile(col_data, [25, 50, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        median = q2 # 값
        # 이상치 인덱스
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        # 이상치 위치에 median 넣기
        x_new[outlier_mask, col] = median
    return x_new

# x = replace_outliers_with_median(x)

# 이상치 평균값처리
def replace_outliers_with_mean(x):
    """
    x : 2차원 numpy array (n_samples, n_features)
    각 열별로 이상치(1.5*IQR 밖)를 해당 열의 평균(mean)으로 대체합니다.
    반환값: 이상치 처리 후 배열 (원본 복사본)
    """
    x_new = x.copy()
    for col in range(x.shape[1]):
        col_data = x[:, col]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mean = np.mean(col_data)
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        x_new[outlier_mask, col] = mean
    return x_new
# x = replace_outliers_with_mean(x)

def clip_outliers_by_iqr(x):
    """
    x : 2차원 numpy array (n_samples, n_features)
    각 열별로 이상치(1.5*IQR 밖)는 lower_bound 또는 upper_bound로 clip(경계값 대체)합니다.
    반환값: 처리된 배열 (원본 복사본)
    """
    x_new = x.copy()
    for col in range(x.shape[1]):
        col_data = x[:, col]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # 하한/상한으로 clip
        x_new[:, col] = np.clip(col_data, lower_bound, upper_bound)
    return x_new
# x = clip_outliers_by_iqr(x)

outlier_tilt = 2
outiler_what = "처리X"
if outlier_tilt == 1:    # 중위값
    x = replace_outliers_with_median(x)
    outiler_what = "중위값"
elif outlier_tilt == 2:  # 평균
    x = replace_outliers_with_mean(x)
    outiler_what = "평균"
elif outlier_tilt == 3:  # 경계값처치
   x = clip_outliers_by_iqr(x)
   outiler_what = "경계처리"

# 서브플롯 (1행 8열)
import matplotlib.pyplot as plt 
fig, axes = plt.subplots(2, 4, figsize=(18, 7))

for idx, ax in enumerate(axes.flat):
    col_data = x[:, idx]
    outlier_loc, iqr, low, up = outlier(col_data)
    ax.boxplot(col_data)
    ax.axhline(up, color='red', label='upper bound')
    ax.axhline(low, color='pink', label='lower bound')
    # ax.set_title(feature_names[idx], fontsize=10)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, train_size=0.8, random_state=seed,
    # stratify=y,
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

model = XGBRegressor(
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
print(f"이상치처리여부 : {outiler_what}")
print('r2_score :', model.score(x_test, y_test))     

# 이상치처리여부 : 이상치처리X
# r2_score : 0.32508665323257446

# 이상치처리여부 : 이상치처리 - 중위값
# r2_score : 0.3263380527496338

# 이상치처리여부 : 이상치처리 - 평균
# r2_score : 0.3280654549598694

# 이상치처리여부 : 이상치처리 - 경계처리
# r2_score : 0.32513147592544556
