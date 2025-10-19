### <<32>>

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)

learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3,4,5,6,7]

# 2. 모델
best_score = 0
best_parameters = ""
for i, lr in enumerate(learning_rate):
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(max_depth=md, learning_rate=lr)
        """
        # 교차검증용 코드
        # 3. 훈련
        scores = cross_val_score(model, x_train, y_train)
        print(lr, '&', md, '조합일때의 평가 acc :', scores)

        # 4. 예측
        y_pred = cross_val_predict(model, x_test, y_test)
        acc = accuracy_score(y_test, y_pred)
        print(lr, ' &', md, '조합일때의 예측 acc :', acc)
        
        if acc > best_score:
            best_score = acc
            # best_parameters = str(lr) + " & " + str(md)
            best_parameters = {'learning_rate' : lr, 'max_depth' : md}
        """
        
        # 단순 훈련, 예측
        model.fit(x_train, y_train)
        score = model.score(x_test,y_test)
        
        print(f'{i+1}, {j+1}번째 수행(score, parameters)', score, lr, md)
        if score > best_score:
            best_score = score
            best_parameters = {'learning_rate' : lr, 'max_depth' : md}
            print(f'갱신된 best score :', best_score)
            print(f'갱신된 lr & md :', lr, '&', md)
        print('=================================')

        
print("최고 점수 : {:.2f}".format(best_score))
print("최적 매개변수 :", best_parameters)
# 최고 점수 : 0.96
# 최적 매개변수 : {'learning_rate': 0.1, 'max_depth': 5}
"""
1, 1번째 수행(score, parameters) 0.9444444444444444 0.1 3
갱신된 best score : 0.9444444444444444
갱신된 lr & md : 0.1 & 3
=================================
1, 2번째 수행(score, parameters) 0.9527777777777777 0.1 4
갱신된 best score : 0.9527777777777777
갱신된 lr & md : 0.1 & 4
=================================
1, 3번째 수행(score, parameters) 0.9611111111111111 0.1 5
갱신된 best score : 0.9611111111111111
갱신된 lr & md : 0.1 & 5
=================================
1, 4번째 수행(score, parameters) 0.9583333333333334 0.1 6
=================================
1, 5번째 수행(score, parameters) 0.9583333333333334 0.1 7
=================================
2, 1번째 수행(score, parameters) 0.9305555555555556 0.05 3
=================================
2, 2번째 수행(score, parameters) 0.95 0.05 4
=================================
2, 3번째 수행(score, parameters) 0.95 0.05 5
=================================
2, 4번째 수행(score, parameters) 0.95 0.05 6
=================================
2, 5번째 수행(score, parameters) 0.9472222222222222 0.05 7
=================================
3, 1번째 수행(score, parameters) 0.8944444444444445 0.01 3
=================================
3, 2번째 수행(score, parameters) 0.9138888888888889 0.01 4
=================================
3, 3번째 수행(score, parameters) 0.9111111111111111 0.01 5
=================================
3, 4번째 수행(score, parameters) 0.9138888888888889 0.01 6
=================================
3, 5번째 수행(score, parameters) 0.9138888888888889 0.01 7
=================================
4, 1번째 수행(score, parameters) 0.8888888888888888 0.005 3
=================================
4, 2번째 수행(score, parameters) 0.8972222222222223 0.005 4
=================================
4, 3번째 수행(score, parameters) 0.9027777777777778 0.005 5
=================================
4, 4번째 수행(score, parameters) 0.9055555555555556 0.005 6
=================================
4, 5번째 수행(score, parameters) 0.9055555555555556 0.005 7
=================================
5, 1번째 수행(score, parameters) 0.8472222222222222 0.001 3
=================================
5, 2번째 수행(score, parameters) 0.8694444444444445 0.001 4
=================================
5, 3번째 수행(score, parameters) 0.8777777777777778 0.001 5
=================================
5, 4번째 수행(score, parameters) 0.8777777777777778 0.001 6
=================================
5, 5번째 수행(score, parameters) 0.8777777777777778 0.001 7
=================================   
"""