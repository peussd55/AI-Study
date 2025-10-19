### <<57>>

import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=777,
    # stratify=y,
)

parameters = [
    {'xgb__n_estimators' : [100, 200], 'xgb__max_depth' : [5, 6, 10], 'xgb__learning_rate' : [0.05, 0.1], 'xgb__subsample': [0.8, 1.0], 'xgb__colsample_bytree': [0.8, 1.0]},
    {'xgb__learning_rate' : [0.05, 0.1], 'xgb__subsample': [0.8, 1.0], 'xgb__colsample_bytree': [0.8, 1.0]},
    {'xgb__subsample': [0.8, 1.0], 'xgb__colsample_bytree': [0.8, 1.0]},
    {'xgb__colsample_bytree': [0.8, 1.0]},
]


# 2. 모델
pipe = Pipeline([('std', StandardScaler()), ('xgb', XGBRegressor(tree_method='hist'))])

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)
print("best_params_:", model.best_params_)
print("best_score_(cv):", model.best_score_)
print("test score:", model.score(x_test, y_test))
# model.score :  0.37907539188755235
# best_params_: {'xgb__colsample_bytree': 0.8, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 5, 'xgb__n_estimators': 100, 'xgb__subsample': 1.0}
# best_score_(cv): 0.3898857952508804
# test score: 0.37907539188755235
# r2_score : 0.37907539188755235

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('r2_score :', acc)