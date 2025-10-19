### <<57>>

import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import warnings; warnings.filterwarnings("ignore", message=".does not have valid feature names.")

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=777,
    stratify=y,
)

parameters = [
{
'lgb__n_estimators': [100],
'lgb__max_depth': [-1, 6, 10],
'lgb__num_leaves': [31], 'lgb__learning_rate': [0.05, 0.1],
'lgb__subsample': [0.8, 1.0],
'lgb__colsample_bytree': [0.8, 1.0],
},
{
'lgb__learning_rate': [0.05, 0.1],
'lgb__subsample': [0.8, 1.0],
'lgb__colsample_bytree': [0.8, 1.0],
},
{
'lgb__subsample': [0.8, 1.0],
'lgb__colsample_bytree': [0.8, 1.0],
},
{
'lgb__colsample_bytree': [0.8, 1.0],
},
]


# 2. 모델
pipe = Pipeline([('std', StandardScaler()), ('lgb', LGBMClassifier(verbose=-1))])

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)
print("best_params_:", model.best_params_)
print("best_score_(cv):", model.best_score_)
print("test score:", model.score(x_test, y_test))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)
# acc : 0.9824561403508771