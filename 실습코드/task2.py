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
print("\n--- 데이터 집합 정보 ---")
df.info()

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
# 특성(X)과 목표값(y)을 분리합니다. (MPG 값을 제외한 나머지를 학습하여 목표인 MPG를 구함)
X = df.drop('MPG', axis=1) 
y = df['MPG']               

# 훈련 데이터와 테스트 데이터를 8:2 비율로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77) 

# 4. 선형 회귀 모델 학습
model = LinearRegression()
# 훈련 데이터를 사용하여 모델을 학습시킵니다.
model.fit(X_train, y_train)

# 5. 선형 회귀 모델 평가
# 테스트 데이터로 예측을 수행합니다.
y_pred = model.predict(X_test)
# 평균 제곱 오차(MSE)를 계산하여 모델의 성능을 평가합니다.
mse = mean_squared_error(y_test, y_pred)
print(f"\n모델의 평균 제곱 오차(MSE): {mse:.2f}")

# 학습된 W와 b를 확인합니다.
print(f"W: {model.coef_}")
print(f"b: {model.intercept_}")

# 6. 예측 결과 및 오차 시각화
plt.figure(figsize=(10, 10))
# 산점도를 그립니다. x축은 예측값, y축은 실제값입니다.
sns.scatterplot(x=y_pred, y=y_test, alpha=0.5)

# 이상적인 예측선 그리기 (y = x)
# 예측값과 실제값이 완벽히 동일할 경우의 직선입니다.
# 이 선에 가까울수록 모델의 성능이 좋다고 할 수 있습니다.
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r', lw=2, label="Ideal (y = x)")

# 데이터의 추세선
# 현재 데이터의 예측 경향을 보여주는 직선입니다.
sns.regplot(x=y_pred, y=y_test, scatter=False, line_kws={'color': 'blue', 'linestyle': '--', 'linewidth': 2}, label='Predicted')

plt.title("GT vs. Preds (MPG)")
plt.xlabel("Model Prediction")
plt.ylabel("Ground Truth")
plt.legend()
plt.grid(True)
plt.axis('equal')   # x축과 y축의 스케일을 동일하게 설정합니다.
plt.show()
