### <<30>>

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=99, train_size=0.8, stratify=y,
)

parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 1,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 3377,
            #   'verbose' : 0,
            }
# 2. 모델
# 모델 파라미터 넣는 법:
# 1. XGBClassifier()에 입력(직접 또는 **문법이용)
# 2. set_params()에 입력(직접 또는 **문법이용)

model = XGBClassifier(
    # **parameters,
    # n_estimators = 1000,
)

# 3.컴파일, 훈련
model.set_params(
                **parameters,
                 early_stopping_rounds=10,  # fit에서 eval_set해야함
                 )

model.fit(x_train, y_train,
        eval_set = [(x_test, y_test)],
        verbose = 10,
          )

results = model.score(x_test, y_test)   # score : tensorflow의 evaluate에 대응
print('최종점수 :', results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc :', acc)

path = './_save/m01_job/'
import pickle
# joblib.dump(model, path + 'm01_joblib_save.joblib')
pickle.dump(model, open(path + 'm02_pickle_save.pickle', 'wb'))  # 확장자는 아무렇게 해도됨. wb : write binary

# 최종점수 : 0.9473684210526315
# acc : 0.9473684210526315