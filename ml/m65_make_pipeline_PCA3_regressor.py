### <<57>>

import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
import warnings
# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 1. 데이터 로딩 함수 정의
def load_fetch_california_housing():
    """1. 유방암 데이터셋 로드"""
    datasets = fetch_california_housing()
    return datasets.data, datasets.target, "California Housing"

def load_diabetes():
    """2. 당뇨병 데이터셋 로드"""
    datasets = fetch_california_housing()
    return datasets.data, datasets.target, "Diabetes"


def load_dacon_ddareung():
    """3. 데이콘 따릉이 데이터셋 로드"""
    try:
        path = './_data/dacon/따릉이/'          # 시스템 경로에서 시작.

        train_csv =  pd.read_csv(path + 'train.csv', index_col=0)     # 0번컬럼을 인덱스컬럼으로 지정
        test_csv = pd.read_csv(path + 'test.csv', index_col=0)  
        submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

        # 결측치 처리
        train_csv = train_csv.fillna(train_csv.mean())
        test_csv = test_csv.fillna(test_csv.mean())

        x = train_csv.drop(['count'], axis=1)   # count 컬럼 제거
        y = train_csv['count']                  # 타겟 변수
        return x.values, y.values, "Dacon DDareung"
    except FileNotFoundError:
        print("Warning: Dacon DDareung dataset not found. Skipping.")
        return None, None, "Dacon DDareung"

def load_kaggle_bike():
    """3. 캐글 바이크 데이터셋 로드"""
    try:
        path = './_data/kaggle/bike/'
        # 맨 앞, 공통 컬럼 datetime을 인덱스 컬럼으로 지정
        train_csv = pd.read_csv(path + 'train.csv', index_col=0)
        test_csv = pd.read_csv(path + 'test.csv', index_col=0)
        submission_csv = pd.read_csv(path + 'sampleSubmission.csv')
        x = train_csv.drop(['casual', 'registered', 'count'], axis=1)   
        y = train_csv['count']
        return x.values, y.values, "Kaggle Bike"
    except FileNotFoundError:
        print("Warning: Kaggle Bike dataset not found. Skipping.")
        return None, None, "Kaggle Bike"


# 데이터셋 로더 리스트
dataset_loaders = [
    load_fetch_california_housing,
    load_diabetes,
    load_dacon_ddareung,
    load_kaggle_bike,
]

# 스케일러 리스트 (이름, 객체)
scalers = [
    ("MinMaxScaler", MinMaxScaler()),
    ("StandardScaler", StandardScaler()),
    ("RobustScaler", RobustScaler())
]

# 모델 리스트 (이름, 객체)
models = [
    ("RandomForest", RandomForestRegressor(random_state=777)),
    ("XGBoost", XGBRegressor(random_state=777)),
    ("CatBoost", CatBoostRegressor(random_state=777, verbose=0)),
    ("LightGBM", LGBMRegressor(random_state=777, verbosity=-1))
]

# 3. 메인 실행 루프
results_list = []
RANDOM_STATE = 777

for loader in dataset_loaders:
    # 데이터 로드
    x, y, dataset_name = loader()
    
    # 데이터셋 로드 실패 시 건너뛰기
    if x is None:
        continue

    print(f"\n{'='*20} Dataset: {dataset_name} {'='*20}")
    print(f"Data Shape: x={x.shape}, y={y.shape}")
    
    # 데이터 분할
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=RANDOM_STATE, 
        # stratify=y
    )

    # 스케일러와 모델 조합으로 반복
    for scaler_name, scaler in scalers:
        for model_name, model in models:
            
            # 파이프라인 생성
            pipeline = make_pipeline(PCA(n_components=8), scaler, model)
            
            try:
                # 훈련
                pipeline.fit(x_train, y_train)
                
                # 평가
                y_predict = pipeline.predict(x_test)
                r2 = r2_score(y_test, y_predict)
                
                # 결과 출력 및 저장
                print(f"  - [{scaler_name} + {model_name}] r2: {r2:.4f}")
                results_list.append({
                    "Dataset": dataset_name,
                    "Scaler": scaler_name,
                    "Model": model_name,
                    "r2_score": r2
                })
                
            except Exception as e:
                print(f"  - [{scaler_name} + {model_name}] ERROR: {e}")
                results_list.append({
                    "Dataset": dataset_name,
                    "Scaler": scaler_name,
                    "Model": model_name,
                    "r2_score": None # 에러 발생 시 None으로 기록
                })
                
# 4. 최종 결과 요약
print(f"\n\n{'='*20} Overall Results Summary {'='*20}")
results_df = pd.DataFrame(results_list)
print(results_df)

# 데이터셋별 최고 성능 조합 출력
print("\n--- Best Combination for Each Dataset ---")
best_results = results_df.loc[results_df.groupby("Dataset")["r2_score"].idxmax()]
print(best_results)

"""
==================== Overall Results Summary ====================
               Dataset          Scaler         Model  r2_score
0   California Housing    MinMaxScaler  RandomForest  0.757604
1   California Housing    MinMaxScaler       XGBoost  0.766609
2   California Housing    MinMaxScaler      CatBoost  0.785244
3   California Housing    MinMaxScaler      LightGBM  0.774020
4   California Housing  StandardScaler  RandomForest  0.757593
5   California Housing  StandardScaler       XGBoost  0.765534
6   California Housing  StandardScaler      CatBoost  0.786443
7   California Housing  StandardScaler      LightGBM  0.774082
8   California Housing    RobustScaler  RandomForest  0.757593
9   California Housing    RobustScaler       XGBoost  0.765534
10  California Housing    RobustScaler      CatBoost  0.786443
11  California Housing    RobustScaler      LightGBM  0.772735
12            Diabetes    MinMaxScaler  RandomForest  0.757604
13            Diabetes    MinMaxScaler       XGBoost  0.766609
14            Diabetes    MinMaxScaler      CatBoost  0.785244
15            Diabetes    MinMaxScaler      LightGBM  0.774020
16            Diabetes  StandardScaler  RandomForest  0.757593
17            Diabetes  StandardScaler       XGBoost  0.765534
18            Diabetes  StandardScaler      CatBoost  0.786443
19            Diabetes  StandardScaler      LightGBM  0.774082
20            Diabetes    RobustScaler  RandomForest  0.757593
21            Diabetes    RobustScaler       XGBoost  0.765534
22            Diabetes    RobustScaler      CatBoost  0.786443
23            Diabetes    RobustScaler      LightGBM  0.772735
24      Dacon DDareung    MinMaxScaler  RandomForest  0.640049
25      Dacon DDareung    MinMaxScaler       XGBoost  0.646815
26      Dacon DDareung    MinMaxScaler      CatBoost  0.667394
27      Dacon DDareung    MinMaxScaler      LightGBM  0.634771
28      Dacon DDareung  StandardScaler  RandomForest  0.640049
29      Dacon DDareung  StandardScaler       XGBoost  0.636083
30      Dacon DDareung  StandardScaler      CatBoost  0.661861
32      Dacon DDareung    RobustScaler  RandomForest  0.640049
33      Dacon DDareung    RobustScaler       XGBoost  0.646815
34      Dacon DDareung    RobustScaler      CatBoost  0.667394
35      Dacon DDareung    RobustScaler      LightGBM  0.638396
36         Kaggle Bike    MinMaxScaler  RandomForest  0.287646
37         Kaggle Bike    MinMaxScaler       XGBoost  0.269596
38         Kaggle Bike    MinMaxScaler      CatBoost  0.339305
39         Kaggle Bike    MinMaxScaler      LightGBM  0.338414
40         Kaggle Bike  StandardScaler  RandomForest  0.287666
41         Kaggle Bike  StandardScaler       XGBoost  0.269596
42         Kaggle Bike  StandardScaler      CatBoost  0.338264
43         Kaggle Bike  StandardScaler      LightGBM  0.336651
44         Kaggle Bike    RobustScaler  RandomForest  0.287568
45         Kaggle Bike    RobustScaler       XGBoost  0.269596
46         Kaggle Bike    RobustScaler      CatBoost  0.338264
47         Kaggle Bike    RobustScaler      LightGBM  0.335025

--- Best Combination for Each Dataset ---
               Dataset          Scaler     Model  r2_score
6   California Housing  StandardScaler  CatBoost  0.786443
26      Dacon DDareung    MinMaxScaler  CatBoost  0.667394
18            Diabetes  StandardScaler  CatBoost  0.786443
38         Kaggle Bike    MinMaxScaler  CatBoost  0.339305
"""
