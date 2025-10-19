### <<31>>
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 8282,
)

# scikit-learn 분류모델은 입력데이터가 전부 1차원이어야한다.(y 원핫인코딩 X, but x 데이터 스케일러는 적용가능)
# 2. 모델구성 (전부 분류)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_list = [LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]    # 클래스가 들어가있음

for index, value in enumerate(model_list):
    if value.__name__ == 'LogisticRegression':
        model = value(solver='liblinear')   # 경고로그안뜨게하려고 적용하는거임
    else:
        model = value()
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(f"{value.__name__} :", results)
    
# LinearSVC : 0.9385964912280702
# LogisticRegression : 0.9649122807017544
# DecisionTreeClassifier : 0.9473684210526315
# RandomForestClassifier : 0.9736842105263158
