import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
import platform

# Step 1. 환경 설정: 윈도우 환경에서 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕으로 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지
else:
    # 다른 OS의 경우 (e.g., Mac)
    plt.rcParams['font.family'] = 'AppleGothic'

# Step 2. 데이터 생성: 선형 분리가 불가능한 원형(동심원) 데이터 생성
# make_circles 함수 사용
# factor는 내부 원과 외부 원 사이의 스케일 비율, noise는 데이터의 노이즈 정도를 설정할 수 있음
X, y = make_circles(100, factor=.1, noise=.1, random_state=77)

# Step 3. 모델 비교 설정
# Gamma 값 리스트 (작은, 중간, 큰 Gamma)
gamma_values = [0.1, 1, 10]
# 각 Gamma 값에 해당하는 제목 설정
titles = ['Gamma=0.1 (작음): 완만한 경계 (과소적합 위험)',
          'Gamma=1.0 (중간): 적절한 경계',
          'Gamma=10.0 (큼): 복잡한 경계 (과적합 위험)']

# Step 4. 시각화 준비 (1행 3열의 subplot 생성)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 시각화를 위한 그리드(격자) 설정 (비선형 경계를 부드럽게 그리기 위해 조밀하게 설정)
# 데이터 범위보다 약간 넓게 설정
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# 0.02 간격으로 그리드 생성
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Step 5. Gamma 값에 따른 모델 학습 및 시각화 반복
for gamma, title, ax in zip(gamma_values, titles, axes):
    # 모델 정의 및 학습
    # kernel='rbf'로 설정하고, gamma 값을 변경. C는 고정 (e.g., 1.0).
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0).fit(X, y)
    
    # 그리드 포인트 전체에 대한 예측 결과를 계산함
    # np.c_는 배열을 열 방향으로 연결함 (좌표 생성)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 시각화: 등고선 채우기(contourf) 사용
    # 예측된 클래스에 따라 영역을 색칠함
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    
    # 실제 데이터 포인트를 산점도로 표시
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', s=50)
    
    # (Optional) 결정 경계 선을 명확하게 표시
    # decision_function을 사용하여 거리가 0인 지점(결정 경계)을 실선으로 그림
    Z_dist = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z_dist, colors='k', levels=[0], linestyles=['-'], linewidths=1.5)
    
    # 각 subplot의 설정
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('특성 1')
    ax.set_ylabel('특성 2')

# 전체 그래프의 제목 설정
plt.suptitle('[실습 3] 비선형 SVM (RBF 커널): 파라미터 Gamma의 영향 비교', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
