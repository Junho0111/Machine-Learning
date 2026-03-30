import pandas as pd # tabular 데이터 처리 라이브러리
from sklearn.datasets import load_diabetes              # 공개 데이터 집합 (당뇨병 데이터)
from sklearn.model_selection import train_test_split    # 학습-평가 데이터 분할 라이브러리 함수
from sklearn.linear_model import LinearRegression       # 선형 회귀 모델
from sklearn.metrics import mean_squared_error          # 비용 함수(MSE)
import seaborn as sns           # 데이터 시각화 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리

# 1. 당뇨병 데이터 로드
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['PROGRESS'] = diabetes.target

print("--- 당뇨병 데이터 집합 샘플 ---")
print(df.head())

# 2. 데이터 분할


# 3. 선형 회귀 모델 학습


# 4. 선형 회귀 모델 평가


# 5. 예측 결과 및 오차 시각화

