import numpy as np  # 벡터, 행렬 연산 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리

# 재선형을 위해 난수 시드 설정 (실행할 때 마다 항상 동일한 결과 보장)
np.random.seed(77)

# 1. 단변수 데이터 생성 (x: 공부 시간, y: 시험 점수)
# 0에서 10 사이의 값을 갖는 50개의 데이터 포인트를 생성합니다.
X = 2 * np.random.rand(50, 1)
# y는 4 + 3x + (가우시안 노이즈)의 형태를 가집니다.
# 노이즈는 생성된 데이터가 실제 데이터처럼 보이게 합니다.
y = 4 + 3 * X + np.random.randn(50, 1)

# 2. 데이터 시각화
# 생성된 데이터의 분포를 확인하기 위해 산점도(scatter plot)를 그립니다.
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.title("Uni-variable Regression")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.grid(True)
plt.show()

# 3-1. 단변수 선형 회귀 모델 파라미터 초기화 (가중치 W와 편향 b)
# W와 b를 임의의 값으로 초기화합니다. 학습을 통해 최적의 값으로 업데이트될 것입니다.
W = np.random.randn(1, 1)
b = np.random.randn(1, 1)

# 3-2. 하이퍼파라미터 설정
# 학습률(learning_rate): 경사 하강법에서 이동할 보폭을 결정합니다. (강의자료 20 페이지)
learning_rate = 0.01
# 반복 횟수(n_iterations): 경사 하강법을 몇 번 반복할지 결정합니다.
n_iterations = 1000

# 데이터 샘플의 개수
m = len(X)
print(f"데이터 샘플의 수: {m}")

# 4. 경사 하강법(Gradient Descent) 구현
# 반복 횟수만큼 학습을 진행합니다.
for iteration in range(n_iterations):
    # 4-1. 선형 가설(Linear hypothesis) 계산: H(X) = WX + b (강의자료 8 페이지)
    # 현재의 W와 b를 사용하여 예측값을 계산합니다.
    predictions = X.dot(W) + b

    # 4-2. 비용 함수(Cost function) 계산: MSE (강의자료 12 페이지)
    # 예측값과 실제값의 차이(오차)를 제곱하여 평균을 냅니다.
    cost = (1 / m) * np.sum((predictions - y) ** 2)
    
    # 4-3. 그래디언트(Gradient) 계산 (강의자료 20 페이지 수식 참고)
    # 비용 함수를 W와 b에 대해 각각 편미분하여 기울기를 구합니다.
    grad_W = (2 / m) * X.T.dot(predictions - y)
    grad_b = (2 / m) * np.sum(predictions - y)

    # 4-4. 파라미터 업데이트 (W와 b 갱신)
    # 계산된 기울기의 반대 방향으로 학습률만큼 파라미터를 이동시킵니다.
    W = W - learning_rate * grad_W
    b = b - learning_rate * grad_b

    # 50번을 반복할 때마다 비용(cost)을 출력하여 학습 과정을 확인합니다.
    if iteration % 50 == 0:
        print(f"Iteration {iteration}, Cost: {cost}, grad_W: {grad_W.item():.4f}, grad_b: {grad_b:.4f}")

# 학습 완료 후 최종 파라미터 출력
print("\n--- 학습 완료 ---")
print(f"최적 가중치 (W): {W.item():.4f}")
print(f"최적 편향 (b): {b.item():.4f}")

# 5. 학습된 선형 회귀선 시각화
plt.figure(figsize=(10, 6))
# 원본 데이터를 산점도로 표시합니다.
plt.scatter(X, y)
# 학습된 W와 b를 사용하여 회귀선을 그립니다.
# X_new는 0부터 2까지의 범위로 회귀선을 그리기 위한 새로운 x 값입니다.
X_new = np.array([[0], [2]])
y_predict = X_new.dot(W) + b
plt.plot(X_new, y_predict, "r-", linewidth=2, label=f"Predicted H(X)={W.item():.4f}X + {b.item():.4f}")
plt.title("Trained Linear Regression Model")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
