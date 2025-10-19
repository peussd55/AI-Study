### <<41>>

import numpy as np
import random
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['font.family'] = 'Malgun Gothic'   # 한글 안깨지게 적용

# 1. 데이터
random.seed(777)
np.random.seed(777)
x = 2 * np.random.rand(100, 1) -1
print(x.shape)      # (100, 1)
print(np.min(x), np.max(x))     # -0.9852230982722201 0.9991190865361039

# 일반적으로 스케일링 → PolynomialFeatures 순서로 적용
# 반대로하면 항이 매우커서 정규화효과 떨어짐
y = 3*x**2 + 2*x + 1 + np.random.rand(100, 1) # y = 3x^2 + 2x + 1 + 노이즈

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf.shape) 
print(x_pf)

# 2. 모델
model = LinearRegression()
model2 = LinearRegression()

# 3. 훈련
model.fit(x, y)
model2.fit(x_pf, y)

# 4. 원래 데이터 그리기
plt.scatter(x, y, color='blue', label='Original Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression 예제')
# plt.show()

# 다항식 회귀 그래프 그리기
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
x_test_pf = pf.transform(x_test)
y_plot = model.predict(x_test)
y_plot_pf = model2.predict(x_test_pf)
plt.plot(x_test, y_plot, color='red', label='기냥')
plt.plot(x_test, y_plot_pf, color='green', label='Polynomial Regression')
# x_test_pf 하면 각 컬럼 에 대한 녹색 그래프 2개나옴

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 가상 데이터 (비선형 관계)
np.random.seed(42)
x1 = np.random.uniform(-2, 2, 50)
x2 = np.random.uniform(-2, 2, 50)
X = np.column_stack([x1, x2])
y = 1 + 2*x1 + 3*x2 + 2*x1*x2 + 2*x1**2 + 0.5*np.random.randn(50)

# 다항 변환
pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)
model = LinearRegression().fit(X_pf, y)

# 평면 그릴 meshgrid
xx, yy = np.meshgrid(np.linspace(-2,2,30), np.linspace(-2,2,30))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_pf = pf.transform(grid)
zz = model.predict(grid_pf).reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c=y, cmap='viridis')
ax.plot_surface(xx, yy, zz, color='green', alpha=0.4)  # 곡면
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('다항회귀 곡면')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# 예시 데이터 생성
np.random.seed(42)
x1 = np.random.uniform(-2, 2, 50)
x2 = np.random.uniform(-2, 2, 50)
X = np.column_stack([x1, x2])
y = 1 + 2*x1 + 3*x2 + 2*x1*x2 + 2*x1**2 + 0.5*np.random.randn(50) # 비선형 포함

# (1) 선형회귀 모델 (PolynomialFeatures 없이)
model_linear = LinearRegression().fit(X, y)

# (2) 시각화용 meshgrid 생성
xx, yy = np.meshgrid(np.linspace(-2,2,30), np.linspace(-2,2,30))
grid = np.c_[xx.ravel(), yy.ravel()]
zz_linear = model_linear.predict(grid).reshape(xx.shape)

# (3) 3D 산점도 + 선형회귀 평면 플롯
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c=y, cmap='viridis', label='실제 데이터')
ax.plot_surface(xx, yy, zz_linear, color='red', alpha=0.4, label='선형회귀 평면')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('PolynomialFeatures 미적용 선형회귀 평면')
plt.show()
