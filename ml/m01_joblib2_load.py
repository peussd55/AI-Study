### <<30>>

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=99, train_size=0.8, stratify=y,
)

# 2. 모델   # 3. 훈련 - 불러오기
path = './_save/m01_job/'
model = joblib.load(path + 'm01_joblib_save.joblib')

# 4. 평가, 예측
results = model.score(x_test, y_test)   # score : tensorflow의 evaluate에 대응
print('최종점수 :', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

# 최종점수 : 0.9473684210526315
# acc : 0.9473684210526315

