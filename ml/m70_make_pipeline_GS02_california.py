### <<57>>

import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=777,
    # stratify=y,
)

parameters = [
    {'xgbregressor__n_estimators' : [100, 200], 'xgbregressor__max_depth' : [5, 6, 10], 'xgbregressor__learning_rate' : [0.05, 0.1], 'xgbregressor__subsample': [0.8, 1.0], 'xgbregressor__colsample_bytree': [0.8, 1.0]},
    {'xgbregressor__learning_rate' : [0.05, 0.1], 'xgbregressor__subsample': [0.8, 1.0], 'xgbregressor__colsample_bytree': [0.8, 1.0]},
    {'xgbregressor__subsample': [0.8, 1.0], 'xgbregressor__colsample_bytree': [0.8, 1.0]},
    {'xgbregressor__colsample_bytree': [0.8, 1.0]},
]


# 2. 모델
pipe = make_pipeline(StandardScaler(), XGBRegressor())
# pipe = Pipeline([('std', StandardScaler()), ('xgb', XGBRegressor(tree_method='hist'))])

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)
print("best_params_:", model.best_params_)
print("best_score_(cv):", model.best_score_)
print("test score:", model.score(x_test, y_test))
# model.score :  0.8413814663650949

# best_params_: {'xgbregressor__colsample_bytree': 0.8, 'xgbregressor__learning_rate': 0.1, 'xgbregressor__max_depth': 6, 'xgbregressor__n_estimators': 200, 'xgbregressor__subsample': 0.8}
# best_score_(cv): 0.8436676940232312
# test score: 0.8413814663650949
# r2_score : 0.8413814663650949

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('r2_score :', acc)