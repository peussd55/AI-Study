### <<31>>

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_digits, load_wine

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
                load_wine(return_X_y=True),
            ]   # x, y를 반환

model_list = [  # 객체가 들어가있음 : ()
            LinearSVC(),
            LogisticRegression(solver='liblinear'), # 경고로그안뜨게하려고 적용하는거임
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            ]

# for index, model in enumerate(model_list): 또는
for model in model_list:
    print(f"======{type(model).__name__}=====")     # model이 객체로 반환되기때문에 해당 클래스 타입알려면 type(인스턴스).__name__ 사용
    for index, (x, y) in enumerate(data_list):      # for index, value in ~ model.fit(value) 하면 model.fit((x, y))처럼 되버림.
        model.fit(x, y)                     
        results = model.score(x, y)
        print(results)
"""
======LinearSVC=====
0.9666666666666667
0.9156414762741653
0.9493600445186422
0.848314606741573
======LogisticRegression=====
0.96
0.9595782073813708
0.993322203672788
0.9719101123595506
======DecisionTreeClassifier=====
1.0
1.0
1.0
1.0
======RandomForestClassifier=====
1.0
1.0
1.0
1.0
"""        