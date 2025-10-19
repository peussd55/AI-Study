### <<57>>

import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 1. 데이터 로딩 함수 정의
def load_cancer():
    """1. 유방암 데이터셋 로드"""
    datasets = load_breast_cancer()
    return datasets.data, datasets.target, "Breast Cancer"

def load_dacon_diabetes():
    """2. 데이콘 당뇨병 데이터셋 로드"""
    try:
        path = './_data/dacon/diabetes/'
        train_csv = pd.read_csv(path + 'train.csv', index_col=0)
        x = train_csv.drop(['Outcome'], axis=1)
        # 0값을 NaN으로 변환 후 각 열의 중앙값으로 대체
        x = x.replace(0, np.nan).fillna(x.median())
        y = train_csv['Outcome']
        return x.values, y.values, "Dacon Diabetes"
    except FileNotFoundError:
        print("Warning: Dacon Diabetes dataset not found. Skipping.")
        return None, None, "Dacon Diabetes"


def load_kaggle_bank():
    """3. 캐글 은행 이탈 고객 데이터셋 로드"""
    try:
        path = './_data/kaggle/bank/'
        train_csv = pd.read_csv(path + 'train.csv', index_col=0)
        
        # 범주형 데이터 인코딩
        le_geo = LabelEncoder()
        le_gen = LabelEncoder()
        train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
        train_csv['Gender'] = le_gen.fit_transform(train_csv['Gender'])
        
        # 불필요한 피처 제거
        train_csv = train_csv.drop(["CustomerId", "Surname"], axis=1)
        
        x = train_csv.drop(['Exited'], axis=1)
        y = train_csv['Exited']
        return x.values, y.values, "Kaggle Bank"
    except FileNotFoundError:
        print("Warning: Kaggle Bank dataset not found. Skipping.")
        return None, None, "Kaggle Bank"

def load_wine_data():
    """4. 와인 데이터셋 로드"""
    datasets = load_wine()
    return datasets.data, datasets.target, "Wine"

def load_digits_data():
    """5. 숫자 필기체 데이터셋 로드"""
    datasets = load_digits()
    return datasets.data, datasets.target, "Digits"

# 데이터셋 로더 리스트
dataset_loaders = [
    load_cancer,
    load_dacon_diabetes,
    load_kaggle_bank,
    load_wine_data,
    load_digits_data
]

# 스케일러 리스트 (이름, 객체)
scalers = [
    ("MinMaxScaler", MinMaxScaler()),
    ("StandardScaler", StandardScaler()),
    ("RobustScaler", RobustScaler())
]

# 모델 리스트 (이름, 객체)
models = [
    ("RandomForest", RandomForestClassifier(random_state=777)),
    ("XGBoost", XGBClassifier(random_state=777)),
    ("CatBoost", CatBoostClassifier(random_state=777, verbose=0)),
    ("LightGBM", LGBMClassifier(random_state=777, verbosity=-1))
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
        x, y, train_size=0.8, shuffle=True, random_state=RANDOM_STATE, stratify=y
    )

    # 스케일러와 모델 조합으로 반복
    for scaler_name, scaler in scalers:
        for model_name, model in models:
            
            # 파이프라인 생성
            pipeline = make_pipeline(scaler, model)
            
            try:
                # 훈련
                pipeline.fit(x_train, y_train)
                
                # 평가
                y_predict = pipeline.predict(x_test)
                acc = accuracy_score(y_test, y_predict)
                
                # 결과 출력 및 저장
                print(f"  - [{scaler_name} + {model_name}] Accuracy: {acc:.4f}")
                results_list.append({
                    "Dataset": dataset_name,
                    "Scaler": scaler_name,
                    "Model": model_name,
                    "Accuracy": acc
                })
                
            except Exception as e:
                print(f"  - [{scaler_name} + {model_name}] ERROR: {e}")
                results_list.append({
                    "Dataset": dataset_name,
                    "Scaler": scaler_name,
                    "Model": model_name,
                    "Accuracy": None # 에러 발생 시 None으로 기록
                })
                
# 4. 최종 결과 요약
print(f"\n\n{'='*20} Overall Results Summary {'='*20}")
results_df = pd.DataFrame(results_list)
print(results_df)

# 데이터셋별 최고 성능 조합 출력
print("\n--- Best Combination for Each Dataset ---")
best_results = results_df.loc[results_df.groupby("Dataset")["Accuracy"].idxmax()]
print(best_results)

"""
==================== Overall Results Summary ====================
           Dataset          Scaler         Model  Accuracy
0    Breast Cancer    MinMaxScaler  RandomForest  0.973684
1    Breast Cancer    MinMaxScaler       XGBoost  0.991228
2    Breast Cancer    MinMaxScaler      CatBoost  0.982456
3    Breast Cancer    MinMaxScaler      LightGBM  1.000000
4    Breast Cancer  StandardScaler  RandomForest  0.973684
5    Breast Cancer  StandardScaler       XGBoost  0.991228
6    Breast Cancer  StandardScaler      CatBoost  0.982456
7    Breast Cancer  StandardScaler      LightGBM  0.991228
8    Breast Cancer    RobustScaler  RandomForest  0.973684
9    Breast Cancer    RobustScaler       XGBoost  0.991228
10   Breast Cancer    RobustScaler      CatBoost  0.982456
11   Breast Cancer    RobustScaler      LightGBM  0.991228
12  Dacon Diabetes    MinMaxScaler  RandomForest  0.702290
13  Dacon Diabetes    MinMaxScaler       XGBoost  0.755725
14  Dacon Diabetes    MinMaxScaler      CatBoost  0.709924
15  Dacon Diabetes    MinMaxScaler      LightGBM  0.702290
16  Dacon Diabetes  StandardScaler  RandomForest  0.702290
17  Dacon Diabetes  StandardScaler       XGBoost  0.755725
18  Dacon Diabetes  StandardScaler      CatBoost  0.709924
19  Dacon Diabetes  StandardScaler      LightGBM  0.717557
20  Dacon Diabetes    RobustScaler  RandomForest  0.702290
21  Dacon Diabetes    RobustScaler       XGBoost  0.755725
22  Dacon Diabetes    RobustScaler      CatBoost  0.709924
23  Dacon Diabetes    RobustScaler      LightGBM  0.709924
24     Kaggle Bank    MinMaxScaler  RandomForest  0.859272
25     Kaggle Bank    MinMaxScaler       XGBoost  0.862665
26     Kaggle Bank    MinMaxScaler      CatBoost  0.864695
27     Kaggle Bank    MinMaxScaler      LightGBM  0.864544
28     Kaggle Bank  StandardScaler  RandomForest  0.859636
29     Kaggle Bank  StandardScaler       XGBoost  0.861878
30     Kaggle Bank  StandardScaler      CatBoost  0.864695
31     Kaggle Bank  StandardScaler      LightGBM  0.864392
32     Kaggle Bank    RobustScaler  RandomForest  0.859848
33     Kaggle Bank    RobustScaler       XGBoost  0.862332
34     Kaggle Bank    RobustScaler      CatBoost  0.864150
35     Kaggle Bank    RobustScaler      LightGBM  0.864059
36            Wine    MinMaxScaler  RandomForest  0.944444
37            Wine    MinMaxScaler       XGBoost  0.916667
38            Wine    MinMaxScaler      CatBoost  0.972222
39            Wine    MinMaxScaler      LightGBM  1.000000
40            Wine  StandardScaler  RandomForest  0.944444
41            Wine  StandardScaler       XGBoost  0.916667
42            Wine  StandardScaler      CatBoost  0.972222
43            Wine  StandardScaler      LightGBM  1.000000
44            Wine    RobustScaler  RandomForest  0.944444
45            Wine    RobustScaler       XGBoost  0.916667
46            Wine    RobustScaler      CatBoost  0.972222
47            Wine    RobustScaler      LightGBM  1.000000
48          Digits    MinMaxScaler  RandomForest  0.977778
49          Digits    MinMaxScaler       XGBoost  0.972222
50          Digits    MinMaxScaler      CatBoost  0.983333
51          Digits    MinMaxScaler      LightGBM  0.980556
52          Digits  StandardScaler  RandomForest  0.975000
53          Digits  StandardScaler       XGBoost  0.972222
54          Digits  StandardScaler      CatBoost  0.983333
55          Digits  StandardScaler      LightGBM  0.977778
56          Digits    RobustScaler  RandomForest  0.975000
57          Digits    RobustScaler       XGBoost  0.972222
58          Digits    RobustScaler      CatBoost  0.983333
59          Digits    RobustScaler      LightGBM  0.972222

--- Best Combination for Each Dataset ---
           Dataset        Scaler     Model  Accuracy
3    Breast Cancer  MinMaxScaler  LightGBM  1.000000
13  Dacon Diabetes  MinMaxScaler   XGBoost  0.755725
50          Digits  MinMaxScaler  CatBoost  0.983333
26     Kaggle Bank  MinMaxScaler  CatBoost  0.864695
39            Wine  MinMaxScaler  LightGBM  1.000000
"""
