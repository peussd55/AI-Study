### <<42>>

# 60 카피

from sklearn.datasets import fetch_california_housing
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
from sklearn.covariance import EllipticEnvelope

# pandas 컬럼명 불일치경고 무시 (x가 pd.dataframe일때 사용)
import warnings
warnings.filterwarnings('ignore', message='X has feature names, but SelectFromModel was fitted without feature names')

seed = 123
random.seed(seed)
np.random.seed(seed)
metric_name = 'rmse'
verbose = 0

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)
feature_names = datasets.feature_names

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
    print(f"컬럼 {col} ({feature_names[col]})")
    print(f"  1사분위: {np.percentile(col_data, 25):.3f},  3사분위: {np.percentile(col_data, 75):.3f}")
    print(f"  IQR: {iqr:.3f}, 하한: {low:.3f}, 상한: {up:.3f}")
    print(f"  이상치 인덱스: {outlier_loc[0]}\n")

# 이상치처리 중위값 - IQR
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

# 이상치처리 중위값 - EllipticEnvelope
def replace_outliers_with_median_elliptic(x, contamination=0.1):
    """
    x : 2차원 numpy array (n_samples, n_features)
    EllipticEnvelope로 이상치로 판정된 행 전체를 각 컬럼 별 중위값으로 대체
    contamination : 이상치 비율 (0.0 ~ 0.5)
    """
    x_new = x.copy()
    # 모델 적합
    detector = EllipticEnvelope(contamination=contamination)
    detector.fit(x)
    outlier_mask = detector.predict(x) == -1  # 이상치 = -1

    # 각 컬럼별 median
    medians = np.median(x, axis=0)
    # 이상치 행 모든 칼럼을 median으로 대체
    x_new[outlier_mask, :] = medians
    return x_new

# x = replace_outliers_with_median_elliptic(x)

# 이상치처리 평균값 - IQR
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

# 이상치처리 평균 - EllipticEnvelope
def replace_outliers_with_mean_elliptic(x, contamination=0.1):
    """
    x : 2차원 numpy array (n_samples, n_features)
    EllipticEnvelope로 이상치로 판정된 행 전체를 각 컬럼 별 중위값으로 대체
    contamination : 이상치 비율 (0.0 ~ 0.5)
    """
    x_new = x.copy()
    # 모델 적합
    detector = EllipticEnvelope(contamination=contamination)
    detector.fit(x)
    outlier_mask = detector.predict(x) == -1  # 이상치 = -1

    # 각 컬럼별 median
    means = np.mean(x, axis=0)
    # 이상치 행 모든 칼럼을 median으로 대체
    x_new[outlier_mask, :] = means
    return x_new

# x = replace_outliers_with_mean_elliptic(x)

# 이상치처리 경계값처리 - IQR
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

# 이상치처리 경계값처리 - EllipticEnvelope
def clip_outliers_by_elliptic(x, contamination=0.1):
    """
    x : 2차원 numpy array (n_samples, n_features)
    EllipticEnvelope로 이상치 행을 찾고, 각 행의 모든 feature를
    해당 컬럼의 IQR 기준 lower~upper bound로 np.clip 처리.
    """
    x_new = x.copy()
    # 1. EllipticEnvelope로 이상치 탐색
    detector = EllipticEnvelope(contamination=contamination)
    detector.fit(x)
    outlier_rows = detector.predict(x) == -1  # 이상치 행 True

    # 2. 각 컬럼별 하한/상한 산출
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 3. 이상치 행만 컬럼별 clip
    x_new[outlier_rows, :] = np.clip(x_new[outlier_rows, :], lower_bound, upper_bound)

    return x_new
# x = clip_outliers_by_elliptic(x)

outlier_tilt = 31
outiler_what = "처리X"
if outlier_tilt == 10:    # 중위값 
    x = replace_outliers_with_median(x)
    outiler_what = "중위값-IQR"
elif outlier_tilt == 20:  # 평균
    x = replace_outliers_with_mean(x)
    outiler_what = "평균-IQR"
elif outlier_tilt == 30:  # 경계값처치
   x = clip_outliers_by_iqr(x)
   outiler_what = "경계처리-IQR"
elif outlier_tilt == 11:    # 중위값
    x = replace_outliers_with_median_elliptic(x)
    outiler_what = "중위값-Elliptic"
elif outlier_tilt == 21:  # 평균
    x = replace_outliers_with_mean_elliptic(x)
    outiler_what = "평균-Elliptic"
elif outlier_tilt == 31:  # 경계값처치
    x = clip_outliers_by_elliptic(x)
    outiler_what = "경계처리-Elliptic"

# 서브플롯 (1행 8열)
import matplotlib.pyplot as plt 
fig, axes = plt.subplots(2, 4, figsize=(18, 7))

for idx, ax in enumerate(axes.flat):
    col_data = x[:, idx]
    outlier_loc, iqr, low, up = outlier(col_data)
    ax.boxplot(col_data)
    ax.axhline(up, color='red', label='upper bound')
    ax.axhline(low, color='pink', label='lower bound')
    ax.set_title(feature_names[idx], fontsize=10)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
# plt.show()

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

"""
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ 그냥 모델훈련 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


========= XGBRegressor ========
이상치처리여부 : 처리X
r2_score : 0.8159363442715617    

이상치처리여부 : 중위값-IQR
r2_score : 0.8123241814643423

이상치처리여부 : 평균-IQR
r2_score : 0.8122144479916156

이상치처리여부 : 경계처리-IQR
r2_score : 0.8179311955161606

이상치처리여부 : 중위값-Elliptic
r2_score : 0.6897194469459464

이상치처리여부 : 평균-Elliptic
r2_score : 0.6935253364473833

이상치처리여부 : 경계처리-Elliptic
r2_score : 0.8237963592525206 

"""