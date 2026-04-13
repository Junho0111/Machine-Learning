import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import platform

# Step 1. 환경 설정: 윈도우 환경에서 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕으로 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지
else:
    # 다른 OS의 경우 (e.g., Mac)
    plt.rcParams['font.family'] = 'AppleGothic'

# Step 2. 데이터 생성: 비선형 데이터 (사인 함수 + 노이즈)
np.random.seed(77)  # 재현성을 위한 랜덤 시드 설정
# 0부터 5 사이의 랜덤한 X 데이터 40개 생성 및 정렬
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# X에 대한 사인 함수 값 계산 (ravel()은 1차원 배열로 평탄화)
y = np.sin(X).ravel()

# 타겟(y)에 노이즈 추가 (현실 데이터 모사)
# 매 5번째 데이터마다 노이즈를 추가함
y[::5] += 3 * (0.5 - np.random.rand(8))

# Step 3. 모델 정의 및 학습 (SVR)
# SVR (Support Vector Regression) 모델 사용
# kernel='rbf'를 사용하여 비선형 회귀를 수행함
epsilon_val = 0.2   # 엡실론 튜브의 폭(반지름 설정)
C_val = 100 # 비용 (페널티 강도) 설정
# gamma='scale'은 데이터의 분산에 따라 gamma를 자동으로 조정
svr_rbf = SVR(kernel='rbf', C=C_val, gamma='scale', epsilon=epsilon_val)
svr_rbf.fit(X, y)   # 모델 학습

# Step 4. 예측
# 시각화를 위해 0부터 5까지 100개의 촘촘한 X 데이터를 생성함
# [:, np.newaxis]는 1차원 배열을 2차원 열 벡터로 변환함 (sklearn 입력 형식에 맞게 조정)
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]
# 학습된 모델로 예측 수행
y_rbf = svr_rbf.predict(X_plot)

# Step 5. 시각화
plt.figure(figsize=(10, 6))

# 실제 데이터 산점도
plt.scatter(X, y, color='darkorange', label='실제 데이터')

# 예측선 (회귀선)
plt.plot(X_plot, y_rbf, color='navy', lw=2, label='RBF 모델 예측선')

# 엡실론 튜브(Epsilon Tube) 시각화
# 예측선을 기준으로 위(+epsilon)와 아래(-epsilon)에 점선을 그림
plt.plot(X_plot, y_rbf + epsilon_val, color='gray', linestyle='--', label=f'엡실론 튜브 (±{epsilon_val})')
plt.plot(X_plot, y_rbf - epsilon_val, color='gray', linestyle='--')

# 튜브 영역 채우기 (fill_between 사용)
# fill_between을 사용하기 위해 X_plot을 1차원으로 변환(ravel())함
plt.fill_between(X_plot.ravel(), y_rbf - epsilon_val, y_rbf + epsilon_val, color='gray', alpha=0.2)

# 서포트 벡터 표시
# SVR에서도 튜브 경계에 있거나 튜브 밖에 있는 데이터가 서포트 벡터가 됨
# svr_rbf.support_는 서포트 벡터의 인덱스를 반환함
support_vectors_idx = svr_rbf.support_
plt.scatter(X[support_vectors_idx], y[support_vectors_idx], facecolors='none', edgecolors='k', s=100, label='서포트 벡터 (튜브 밖/경계)')

plt.xlabel('데이터 (X)')
plt.ylabel('타겟 (y)')
plt.title(f'[실습 4] 서포트 벡터 회귀 (SVR) 시각화 (C={C_val}, Epsilon={epsilon_val})')
plt.legend()
plt.show()
