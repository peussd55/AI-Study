### <<34>>

# Voting + 베이지안 + cv 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import time

# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333,)

# 2. 모델
bayesian_params = {
    'xgb_n_estimators': (50, 300),
    'xgb_max_depth': (3, 10),
    'xgb_learning_rate': (0.01, 0.3),
    'lgbm_n_estimators': (50, 300),
    'lgbm_num_leaves': (20, 150),
    'lgbm_learning_rate': (0.01, 0.3),
    'cat_iterations': (50, 300),
    'cat_depth': (3, 10),
    'cat_learning_rate': (0.01, 0.3),
}

# 베이지안 최적화용 함수 정의
def voting_cv(xgb_n_estimators, xgb_max_depth, xgb_learning_rate,
              lgbm_n_estimators, lgbm_num_leaves, lgbm_learning_rate,
              cat_iterations, cat_depth, cat_learning_rate):

    xgb_n_estimators = int(round(xgb_n_estimators))
    xgb_max_depth = int(round(xgb_max_depth))
    lgbm_n_estimators = int(round(lgbm_n_estimators))
    lgbm_num_leaves = int(round(lgbm_num_leaves))
    cat_iterations = int(round(cat_iterations))
    cat_depth = int(round(cat_depth))

    xgb = XGBClassifier(
        n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate,
        use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=333
    )
    lgbm = LGBMClassifier(
        n_estimators=lgbm_n_estimators, num_leaves=lgbm_num_leaves, learning_rate=lgbm_learning_rate,
        verbosity=-1, random_state=333
    )
    cat = CatBoostClassifier(
        iterations=cat_iterations, depth=cat_depth, learning_rate=cat_learning_rate,
        verbose=0, random_seed=333
    )

    model = VotingClassifier(
        estimators=[('XGB', xgb), ('LGBM', lgbm), ('CAT', cat)],
        voting='soft'
    )
    
    # cv적용 : 베이지안은 cv옵션없어서 직접 구현해야함. 이거 사용하려면 train_test_split하지말아야할듯. test데이터 써먹을 수있는 코드가 작성안됨.
    cv_scores = []
    for train_idx, val_idx in kfold.split(x_train, y_train):
        X_tr, X_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        # 손실함수 accuracy_score 사용
        acc = accuracy_score(y_val, y_pred)
        cv_scores.append(acc)

    mean_cv_score = np.mean(cv_scores)
    print(f'CV Accuracy: {mean_cv_score}')
    return mean_cv_score

    # 그냥적용
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # acc = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {acc}')
    # return acc

# 베이지안 옵티마이제이션 실행
optimizer = BayesianOptimization(
    f=voting_cv,
    pbounds=bayesian_params,
    random_state=333,
    verbose=2
)

n_iter = 2  # 반복 횟수(시간에 따라 조절)
start = time.time()
optimizer.maximize(init_points=1, n_iter=n_iter)
end = time.time()

print('최적 결과:', optimizer.max)
print(n_iter, '번 걸린 시간 :', round(end - start), '초')

# cv 적용X
# 최적 결과: {'target': 0.8260342598577892, 'params': {'cat_depth': 10.0, 'cat_iterations': 204.80552143429458, 'cat_learning_rate': 0.3, 'lgbm_learning_rate': 0.3, 'lgbm_n_estimators': 50.0, 'lgbm_num_leaves': 150.0, 'xgb_learning_rate': 0.2395502280065108, 'xgb_max_depth': 9.05409668401997, 'xgb_n_estimators': 217.19019135704787}}
# 20 번 걸린 시간 : 995 초

# cv 적용O
# 무지오래걸려서 중단