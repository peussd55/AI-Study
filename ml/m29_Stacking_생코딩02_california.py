# ### <<35>>

# # stacking : 모델객체를 여러개 만들고 모델별 결과를 컬럼으로 생성한뒤 재학습 하는 방법

# import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
# from sklearn.metrics import r2_score
# from sklearn.linear_model import Ridge
# import warnings

# # 1. 데이터
# x, y =fetch_california_housing(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=517,
#     # stratify=y,
# )

# # 2. 모델
# xgb = XGBRegressor()
# rf = RandomForestRegressor()
# cat = CatBoostRegressor(verbose=0)
# lg = LGBMRegressor()

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

#     score = r2_score(y_test, y_test_pred)
#     class_name = model.__class__.__name__
#     print('{0} R2 : {1:.4f}'.format(class_name, score))
#     # XGBRegressor R2 : 0.8281
#     # RandomForestRegressor R2 : 0.8028
#     # CatBoostRegressor R2 : 0.8479
#     # LGBMRegressor R2 : 0.8301
    
# x_train_new = np.array(train_list).T
# print(x_train_new)
# print(x_train_new.shape)    # (16512, 4)

# x_test_new = np.array(test_list).T
# print(x_test_new.shape)     # (4128, 4)

# # 2-2 모델 : stacking
# model2 = Ridge()
# model2.fit(x_train_new, y_train)
# y_pred2 = model2.predict(x_test_new)
# score2 = r2_score(y_test, y_pred2)
# print('스태킹 결과 :', score2)
# # CatBoostRegressor : 0.7863202504412381
# # XGBRegressor : 0.7854018670344692
# # RandomForestRegressor : 0.7823367424721355
# # LGBRegressor : 0.7889441394648725
# # -> 개별모델보다 성능 떨어짐 -> 과적합이 원인?

# 홀드아웃방식(validation 데이터 분리 후 스태킹모델훈련할때 사용)
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
x, y =fetch_california_housing(return_X_y=True)

# 1. 데이터를 3분할: train/validation/test
x_temp, x_test, y_temp, y_test = train_test_split(
    x, y, test_size=0.2, random_state=517
)
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.25, random_state=517  # 전체의 20%를 validation으로
)

print(f"Train: {x_train.shape}, Validation: {x_val.shape}, Test: {x_test.shape}")

# 2. 1차 모델들 학습
models = [
    XGBRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    CatBoostRegressor(verbose=0, random_state=42),
    LGBMRegressor(verbose=0, verbosity=-1, random_state=42)
]

train_preds = []
val_preds = []
test_preds = []

for i, model in enumerate(models):
    # 훈련 데이터로 학습
    model.fit(x_train, y_train)
    
    # 각 데이터셋에 대한 예측
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    
    train_preds.append(train_pred)
    val_preds.append(val_pred)
    test_preds.append(test_pred)
    
    # 개별 성능 확인
    score = r2_score(y_test, test_pred)
    print(f"Model {i+1} R2: {score:.4f}")
    # Model 1 R2: 0.8250
    # Model 2 R2: 0.7980
    # Model 3 R2: 0.8438
    # Model 4 R2: 0.8248

# 3. 스태킹용 데이터 생성
x_train_stack = np.column_stack(train_preds)
x_val_stack = np.column_stack(val_preds)
x_test_stack = np.column_stack(test_preds)

# 4. 메타 모델 학습 (validation 데이터 사용)
meta_model = Ridge(alpha=1.0)   # 스태킹에 적합한 모델(분류) : Ridge
# 부스팅 모델은 비선형관계도 학습하기때문에 최종 스태킹에 사용하면 과적합이 잘 된다. 따라서 스태킹최종모델은 단순한 모델이어야한다.
meta_model.fit(x_val_stack, y_val)

# 5. 최종 예측
final_pred = meta_model.predict(x_test_stack)
stacking_score = r2_score(y_test, final_pred)
print(f"\n스태킹 R2: {stacking_score:.4f}")
# 스태킹 R2: 0.8457