import numpy as np  # 벡터, 행렬 연산 라이브러리
import matplotlib.pyplot as plt # 시각화 라이브러리
from mpl_toolkits.mplot3d import Axes3D # 3차원 시각화 라이브러리

# 재선형을 위해 난수 시드 설정 (실행할 때 마다 항상 동일한 결과 보장)
np.random.seed(77)

# 1. 단변수 데이터 생성 (x: 공부 시간, y: 시험 점수)
# 0에서 10 사이의 값을 갖는 50개의 데이터 포인트를 생성합니다.
X = 2 * np.random.rand(50, 1)
# y는 4 + 3x + (가우시안 노이즈)의 형태를 가집니다.
# 노이즈는 생성된 데이터가 실제 데이터처럼 보이게 합니다.
y = 4 + 3 * X + np.random.randn(50, 1)

# 2. 비용 함수를 계산하기 위한 W, b의 범위 설정
# -5부터 10까지 100개의 점을 생성하여 W와 b의 그리드를 생성
W_range = np.linspace(-5, 10, 100)
b_range = np.linspace(-5, 10, 100)
W_grid, b_grid = np.meshgrid(W_range, b_range)

# 3. 각 (W, b) 조합에 대한 비용 계산
# 모든 (W, b) 쌍에 대해 비용(MSE)을 계산합니다.
cost_grid = np.zeros_like(W_grid)
m = len(X)

for i in range(W_grid.shape[0]):
    for j in range(W_grid.shape[1]):
        w_ij = W_grid[i, j]
        b_ij = b_grid[i, j]
        predictions = X * w_ij + b_ij
        cost = (1 / m) * np.sum((predictions - y) ** 2)
        cost_grid[i, j] = cost

# 4. 경사 하강법을 실행하며 그래디언트 이동경로를 저장합니다.
W = np.random.randn(1, 1)   # 파라미터 재초기화
b = np.random.randn(1, 1)   # 파라미터 재초기화
learning_rate = 0.1
n_iterations = 100

# 학습 과정에서 W, b, cost의 변화를 기록할 리스트 초기화
path_W , path_b, path_cost = [], [], []

for iteration in range(n_iterations):
    predictions = X.dot(W) + b
    cost = (1 / m) * np.sum((predictions - y) ** 2)
    # 경로 저장
    path_W.append(W.item())
    path_b.append(b.item())
    path_cost.append(cost)

    grad_W = (2 / m) * X.T.dot(predictions - y)
    grad_b = (2 / m) * np.sum(predictions - y)
    W = W - learning_rate * grad_W
    b = b - learning_rate * grad_b

# 5. 3D 시각화
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 비용 함수 표면을 그립니다.
ax.plot_surface(W_grid, b_grid, cost_grid, cmap='viridis', alpha=0.6)
# 경사 하강법 경로를 점으로 표시합니다.
ax.scatter(path_W, path_b, path_cost, color='red', s=40, label='Trajectory')

ax.set_title("Gradient Descent Visualization")
ax.set_xlabel("W")
ax.set_ylabel("b")
ax.set_zlabel("Cost")
ax.legend()
plt.show()
