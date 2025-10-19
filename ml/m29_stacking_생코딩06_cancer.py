# ### <<35>>

# # stacking : 모델객체를 여러개 만들고 모델별 결과를 컬럼으로 생성한뒤 재학습 하는 방법

# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from xgboost import XGBRegressor, XGBClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier
# from sklearn.metrics import r2_score, accuracy_score
# from sklearn.linear_model import Ridge, LogisticRegression
# import warnings
# warnings.filterwarnings('ignore')

# # 1. 데이터
# x, y =load_breast_cancer(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=517,
#     stratify=y,
# )

# # 2. 모델
# xgb = XGBClassifier()
# rf = RandomForestClassifier()
# cat = CatBoostClassifier(verbose=0)
# lg = LGBMClassifier(verbose=0, verbosity=-1)

# models = [xgb, rf, cat ,lg]

# train_list = []
# test_list = []

# # 2.1 모델 : 개별성능평가
# for model in models:
#     model.fit(x_train, y_train)
#     y_train_pred = model.predict(x_train)         # y_train_pred만드는 이유? stacking할때 필요한 모델별 컬럼을 생성하기위해
#     y_test_pred = model.predict(x_test)
    
#     train_list.append(y_train_pred)
#     test_list.append(y_test_pred)

#     score = accuracy_score(y_test, y_test_pred)
#     class_name = model.__class__.__name__
#     print('{0} R2 : {1:.8f}'.format(class_name, score))
#     # XGBClassifier R2 : 0.9649
#     # RandomForestClassifier R2 : 0.9737
#     # CatBoostClassifier R2 : 0.9649
#     # LGBMClassifier R2 : 0.9649
    
# x_train_new = np.array(train_list).T
# print(x_train_new)
# print(x_train_new.shape)    # (16512, 4)

# x_test_new = np.array(test_list).T
# print(x_test_new.shape)     # (4128, 4)

# # 2-2 모델 : stacking
# model2 = LogisticRegression()       # 스태킹에 적합한 모델(분류) : LogisticRegression
# # 부스팅 모델은 비선형관계도 학습하기때문에 최종 스태킹에 사용하면 과적합이 잘 된다. 따라서 스태킹최종모델은 단순한 모델이어야한다.
# model2.fit(x_train_new, y_train)
# y_pred2 = model2.predict(x_test_new)
# score2 = accuracy_score(y_test, y_pred2)
# print('{0}스태킹 결과 :'.format(model2.__class__.__name__), score2)
# # LogisticRegression : 0.9649122807017544
# # -> 개별모델보다 성능 떨어짐 -> 과적합이 원인?


# 홀드아웃방식(validation 데이터 분리 후 스태킹모델훈련할때 사용)
# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from xgboost import XGBRegressor, XGBClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier
# from catboost import CatBoostRegressor, CatBoostClassifier
# from sklearn.metrics import r2_score, accuracy_score
# from sklearn.linear_model import Ridge, LogisticRegression
# import warnings
# warnings.filterwarnings('ignore')

# # 1. 데이터
# x, y =load_breast_cancer(return_X_y=True)

# # 1. 데이터를 3분할: train/validation/test
# x_temp, x_test, y_temp, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=517, stratify=y
# )
# x_train, x_val, y_train, y_val = train_test_split(
#     x_temp, y_temp, test_size=0.25, random_state=517, stratify=y_temp  # 전체의 20%를 validation으로
# )

# # 2. 모델
# models = [
#     XGBClassifier(random_state=42),
#     RandomForestClassifier(random_state=42),
#     CatBoostClassifier(verbose=0, random_state=42),
#     LGBMClassifier(verbose=0, verbosity=-1, random_state=42)
# ]

# train_list = []
# val_list = []
# test_list = []

# # 2.1 모델 : 개별성능평가
# for i, model in enumerate(models):
#     # 훈련 데이터로 학습
#     model.fit(x_train, y_train)
    
#     # 각 데이터셋에 대한 예측
#     y_train_pred = model.predict(x_train)         # y_train_pred만드는 이유? stacking할때 필요한 모델별 컬럼을 생성하기위해
#     y_val_pred = model.predict(x_val)
#     y_test_pred = model.predict(x_test)
    
#     train_list.append(y_train_pred)
#     val_list.append(y_val_pred)
#     test_list.append(y_test_pred)

#     # 개별 성능 확인
#     score = accuracy_score(y_test, y_test_pred)
#     class_name = model.__class__.__name__
#     print('{0} R2 : {1:.8f}'.format(class_name, score))
#     # XGBClassifier R2 : 0.95614035
#     # RandomForestClassifier R2 : 0.94736842
#     # CatBoostClassifier R2 : 0.94736842
#     # LGBMClassifier R2 : 0.95614035
    
# x_train_new = np.array(train_list).T
# print(x_train_new.shape)    # (341, 4)

# x_val_new = np.array(val_list).T
# print(x_val_new.shape)      # (114, 4)

# x_test_new = np.array(test_list).T
# print(x_test_new.shape)     # (114, 4)

# # 2-2 모델 : stacking
# model2 = LogisticRegression()       # 스태킹에 적합한 모델(분류) : LogisticRegression
# # 부스팅 모델은 비선형관계도 학습하기때문에 최종 스태킹에 사용하면 과적합이 잘 된다. 따라서 스태킹최종모델은 단순한 모델이어야한다.
# model2.fit(x_val_new, y_val)
# y_pred2 = model2.predict(x_test_new)
# score2 = accuracy_score(y_test, y_pred2)
# print('{0}스태킹 결과 :'.format(model2.__class__.__name__), score2)
# # LogisticRegression스태킹 결과 : 0.9473684210526315 -> 개별모델보다 성능 떨어짐 -> 데이터가 너무 적어서 데이터소실로인해 성능이 떨어지는 것 같음


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

# 2. 모델
models = [
    XGBClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    CatBoostClassifier(verbose=0, random_state=42),
    LGBMClassifier(verbose=0, verbosity=-1, random_state=42)
]

# 3. KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. 메타 특성 초기화 (전체 데이터 크기 x 모델 수)
meta_features = np.zeros((len(x), len(models)))

print("=== Cross-Validation 기반 메타 특성 생성 ===")
# 5. 각 모델별로 KFold CV를 통해 메타 특성 생성
for i, model in enumerate(models):
    print(f"Processing {model.__class__.__name__}...")
    cv_scores = []
    
    for train_idx, val_idx in kf.split(x):
        # 폴드별 train/validation 분할
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 모델 학습
        model.fit(x_train, y_train)
        
        # validation 데이터 예측 (메타 특성으로 사용)
        val_pred = model.predict(x_val)
        meta_features[val_idx, i] = val_pred
        
        # CV 성능 확인
        cv_score = accuracy_score(y_val, val_pred)
        cv_scores.append(cv_score)
    
    # 모델별 CV 평균 성능 출력
    print(f'{model.__class__.__name__} CV Accuracy: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores)*2:.6f})')

print(f"\n메타 특성 shape: {meta_features.shape}")

# 6. 메타 모델 학습을 위한 train/test 분할
x_train_meta, x_test_meta, y_train_meta, y_test_meta = train_test_split(
    meta_features, y, test_size=0.2, random_state=517, stratify=y
)

print(f"메타 모델 Train shape: {x_train_meta.shape}")
print(f"메타 모델 Test shape: {x_test_meta.shape}")

# 7. 메타 모델 학습
meta_model = LogisticRegression(random_state=42)
meta_model.fit(x_train_meta, y_train_meta)

# 8. 최종 예측 및 평가
final_pred = meta_model.predict(x_test_meta)
final_score = accuracy_score(y_test_meta, final_pred)

print(f"\n=== 최종 결과 ===")
print(f'{meta_model.__class__.__name__} 스태킹 결과: {final_score:.6f}')

# 9. 개별 모델들의 테스트 성능과 비교
print(f"\n=== 개별 모델 vs 스태킹 비교 ===")
for i, model in enumerate(models):
    # 전체 데이터로 재학습 후 테스트
    model.fit(x_train_meta, y_train_meta)  # 메타 특성이 아닌 원본 특성으로 학습하려면 별도 처리 필요
    individual_score = accuracy_score(y_test_meta, x_test_meta[:, i])  # 해당 모델의 메타 특성 사용
    print(f'{model.__class__.__name__}: {individual_score:.6f}')

print(f'Cross-Validation 스태킹: {final_score:.6f}')
# XGBClassifier: 0.964912
# RandomForestClassifier: 0.956140
# CatBoostClassifier: 0.973684
# LGBMClassifier: 0.973684
# Cross-Validation 스태킹: 0.973684
# -> cv방식으로 바꿨음에도 스태킹점수가 개별모드 최고점수와 동등할 뿐 갱신은 못했다.