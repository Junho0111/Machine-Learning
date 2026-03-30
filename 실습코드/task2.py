import pandas as pd # tabular 데이터 처리 라이브러리
from sklearn.model_selection import train_test_split    # 학습-평가 데이터 분할 라이브러리 함수
from sklearn.linear_model import LinearRegression       # 선형 회귀 모델
from sklearn.metrics import mean_squared_error          # 비용 함수(MSE)
import seaborn as sns           # 데이터 시각화 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리

# 1. Auto MPG 데이터 로드 (University of California Irvine Archive)
# 데이터 집합을 URL로 바로 로드합니다.
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# 컬럼(속성) 이름은 별도로 지정해줍니다.
# MPG (목표값): 연비 (Miles Per Gallon, 갤런당 마일 수)
# Displacement: 배기량 (엔진의 크기)
# Horsepower: 마력 (엔진의 출력)
# Weight: 차량 무게
# Acceleration: 가속 성능 (0 to 60mph까지 걸리는 시간)
# Model Year: 출시 연도
# Origin_1: 제조국-미국산 (원-핫 인코딩)
# Origin_2: 제조국-유럽 (원-핫 인코딩)
# Origin_3: 제조국-일본 (원-핫 인코딩)
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car Name']
df = pd.read_csv(url, names=column_names, na_values='?', sep=r'\s+')

print("--- Auto MPG 데이터 집합 샘플 ---")
print(df.head())

# 2. 데이터 전처리
# 2-1. 결측치 확인 및 처리
# Horsepower 속성에 '?'로 표시된 결측치가 있습니다. 이를 제거합니다.
print("\n--- 전처리: 결측치 확인 ---")
print(df.isna().sum())
df = df.dropna()

# 2-2. 불필요한 속성 제거
# 자동차 이름(Car Name)은 모델 학습에 사용되지 않으므로 제거합니다.
df = df.drop('Car Name', axis=1)

# 2-3. 범주형 데이터 처리 ('Origin' 속성)
# Origin 속성은 1, 2, 3으로 표기되어 있으나 실제로는 미국, 유럽, 일본을 의미하는 범주형 데이터입니다.
# 원-핫 인코딩을 통해 모델이 이해할 수 있는 형태로 변환합니다.
df = pd.get_dummies(df, columns=['Origin'], prefix='Origin')
print('\n--- 전처리 후 데이터 집합 샘플 ---')
print(df.head())

# 3. 데이터 분할


# 4. 선형 회귀 모델 학습


# 5. 선형 회귀 모델 평가


# 6. 예측 결과 및 오차 시각화

