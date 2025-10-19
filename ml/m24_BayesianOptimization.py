### <<34>>

# 베이지안 옵티마아제이션 :  GridSearch, RandomSearch처럼 최적의 파라미터 조합을 찾는 튜닝기법중 중 하나.
# 베이지안 특징 : 모델객체를 반환하지않고 최종결과만 반환하기때문에 로직을 블랙박수 함수에서 다 작성해야한다.

param_bonus = {'x1' : (-1,5),
               'x2' : (0,4)
               }
# 최적화할 변수 x1의 범위: -1 ~ 5 / 최적화할 변수 x2의 범위: 0 ~ 4

def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 + 10
# = -x1^2 -(x2-2)^2 + 10
# x1은 0, x2는 2일때 최대값 도출

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,         # 최적화할 함수. 블랙박스 함수
    pbounds = param_bonus,  # 변수의 범위
    random_state=333,
)

optimizer.maximize(init_points=5, n_iter=20)    # maximize : y_function이 최대값이 되는것을 찾는 함수 / init_points : 초기 반복횟수 / n_iter : 추가 반복횟수
# 로그에서 값이 갱신되면 색상이 생김
# 가우시안 프로세스 알고리즘으로 찾음

print(optimizer.max)    # {'target': 9.999999030311658, 'params': {'x1': -0.00027579246465858425, 'x2': 1.999054681610145}}
# maximize: (최대값을 찾는)탐색을 "실행"하는 함수
# max: 탐색 결과 중 "최고값"을 저장하는 속성
