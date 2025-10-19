### <<57>>

import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=777,
    stratify=y,
)

parameters = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [5, 6, 10], 'rf__min_samples_leaf' : [3, 10]},  # 12
    {'rf__max_depth' : [6, 8, 10, 12], 'rf__min_samples_leaf' : [3, 5, 7, 10]},     # 16
    {'rf__min_samples_leaf' : [3, 5, 7, 9], 'rf__min_samples_split' : [2,3,5,10]},  # 16
    {'rf__min_samples_split' : [2,3,5,6]},  # 1
]

# 2. 모델
pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score : ', results)
print("best_params_:", model.best_params_)
print("best_score_(cv):", model.best_score_)
print("test score:", model.score(x_test, y_test))
# model.score :  0.9333333333333333
# best_params_: {'rf__max_depth': 5, 'rf__min_samples_leaf': 3, 'rf__n_estimators': 100}
# best_score_(cv): 0.975
# test score: 0.9333333333333333
# accuracy_score : 0.9333333333333333

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)