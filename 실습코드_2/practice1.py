import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import platform

# Step 1. 환경 설정: 윈도우 환경에서 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'   # 맑은 고딕으로 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지
else:
    # 다른 OS의 경우 (e.g., Mac)
    plt.rcParams['font.family'] = 'AppleGothic'

# Step 2. 데이터 생성: 선형 분리 가능한 데이터
# 40개의 샘플, 2개의 클래스(centers=2), 표준편자 0.6으로 설정하여 완벽히 분리되도록 함
X, y = make_blobs(n_samples=40, centers=2, random_state=77, cluster_std=0.6)

# Step 3. 모델 정의 및 학습: 하드 마진 SVM (C를 매우 크게 설정해서 하드 마진처럼 동작하도록 함)
# kernel='linear'로 설정하여 선형 SVM을 선택
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)   # 모델 학습

# Step 4. 시각화 설정
plt.figure(figsize=(8, 6))
# 데이터 포인트를 산점도로 표시 (c=y는 클래스 레이블에 따라 색상 구분)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 결정 경계 및 마진 시각화를 위한 설정
ax = plt.gca()  # 현재 Axes 객체 가져오기
xlim = ax.get_xlim()    # X축 범위
ylim = ax.get_ylim()    # Y축 범위

# 그래프를 그리기 위한 그리드(격자) 생성
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)    # 2차원 그리드 생성
# 그리드 포인트를 (x, y) 쌍의 배열로 변환
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# 결정 함수(decision_function)를 사용하여 각 지점의 거리 계산
# Z는 결정 경계로부터의 거리를 나타냄
Z = clf.decision_function(xy).reshape(XX.shape)

# Step 5. 결정 경계와 마진 그리기
# 등고선 (contour)으로 표시
# levels=[-1, 0, 1]: 거리가 -1, 0, 1인 지점을 그림
# 0은 결정 경계(실선), -1과 1은 마진 경계(점선)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Step 6. 서포트 벡터 표시
# clf.support_vectors_는 모델이 찾은 서포트 벡터들의 좌표에 해당함
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
          linewidth=1, facecolors='none', edgecolors='k', label='서포트 벡터')

plt.title('[실습 1] 하드 마진 SVM 시각화 (C=1000)')
plt.xlabel('특성 1')
plt.ylabel('특성 2')
plt.legend()
plt.show()
