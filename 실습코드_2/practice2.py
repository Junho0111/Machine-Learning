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
    
# Step 2. 데이터 생성: 약간 겹치는 데이터 생성
# 표준편차(cluster_std)를 2.0로 설정하여 데이터가 겹치도록 합니다 (현실 데이터 모사)
X, y = make_blobs(n_samples=100, centers=2, random_state=77, cluster_std=2.0)

# Step 3. 모델 비교 설정
# C 값 리스트 (작은 C와 큰 C)
C_values = [0.1, 100.0]
# 각 C 값에 해당하는 제목 설정
titles = ['C=0.1 (작은 C): 넓은 마진, 많은 오분류 허용',
          'C=100.0 (큰 C): 좁은 마진, 적은 오분류 허용']

# Step 4. 시각화 준비 (1행 2열의 subplot 생성)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Step 5. C 값에 따른 모델 학습 및 시각화 반복
# zip을 사용하여 C 값, 제목, 그리고 subplot 객체를 순회함
for C, title, ax in zip(C_values, titles, axes):
    # 모델 정의 및 학습
    # 선형 커널을 사용하고 C 값을 변경하여 학습함
    clf = svm.SVC(kernel='linear', C=C).fit(X, y)
    
    # 데이터 산점도 표시
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    
    # 결정 경계 및 마진 시각화 (실습 1과 동일한 로직)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # 결정 경계와 마진 표시
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # 서포트 벡터 표시
    # 소프트 마진에서는 마진 경계에 있는 데이터 뿐만 아니라, 마진을 위반한 데이터도 서포트 벡터가 됨
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    
    # 각 subplot의 제목 및 레이블 설정
    ax.set_title(title)
    ax.set_xlabel('특성 1')
    ax.set_ylabel('특성 2')
    
# 전체 그래프의 제목 설정
plt.suptitle('[실습 2] 소프트 마진 SVM: 파라미터 C의 영향 비교', fontsize=16)
# 레이아웃 조정 (suptitle과 subplot 제목이 겹치지 않도록 조정)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
