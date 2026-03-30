import pandas as pd # tabular 데이터 처리 라이브러리
from sklearn.datasets import fetch_california_housing   # 공개 데이터 집합 (캘리포니아 주택 가격)
from sklearn.model_selection import train_test_split    # 학습-평가 데이터 분할 라이브러리 함수
from sklearn.linear_model import LinearRegression       # 선형 회귀 모델
from sklearn.metrics import mean_squared_error          # 비용 함수(MSE)
import seaborn as sns           # 데이터 시각화 라이브러리
import matplotlib.pyplot as plt # 데이터 시각화 라이브러리

# 1. 캘리포니아 주택 가격 데이터 로드
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
# 'PRICE' 열에 목표값(주택 가격) 데이터를 추가합니다.
df['PRICE'] = housing.target

print("--- 캘리포니아 주택 가격 데이터 집합 샘플 ---")
print(df.head())
print("\n--- 데이터 집합 정보 ---")
df.info()

# 2. 학습 / 평가 데이터 부분집합 분할
# 특성(X)과 목표값(y)을 분리합니다.
X = df.drop('PRICE', axis=1)
y = df['PRICE']
# 훈련 데이터와 테스트 데이터를 8:2 비율로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# 3. 선형 회귀 모델 학습
# 선형 회귀 모델을 생성합니다.
model = LinearRegression()
# 훈련 데이터를 사용하여 모델을 학습시킵니다.
model.fit(X_train, y_train)

# 4. 모델 평가
# 테스트 데이터로 예측을 수행합니다.
y_pred = model.predict(X_test)
# 평균 제곱 오차(MSE)를 계산하여 모델의 성능을 평가합니다.
mse = mean_squared_error(y_test, y_pred)
print(f"\n모델의 평균 제곱 오차(MSE): {mse:.2f}")

# 학습된 W와 b를 확인합니다.
print(f"W: {model.coef_}")
print(f"b: {model.intercept_}")

# 5. 예측 결과 및 오차 시각화
plt.figure(figsize=(10, 10))
# 5-1. 산점도를 그립니다. x축은 예측값, y축은 실제값입니다.
sns.scatterplot(x=y_pred, y=y_test, alpha=0.5)

# 5-2. 이상적인 예측선 그리기 (y = x)
# 예측값과 실제값이 완벽히 동일할 경우의 직선입니다.
# 이 선에 가까울수록 모델의 성능이 좋다고 할 수 있습니다.
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r', lw=2, label="Ideal (y = x)")

# 5-3. 데이터의 추세선
# 현재 데이터의 예측 경향을 보여주는 직선입니다.
sns.regplot(x=y_pred, y=y_test, scatter=False, line_kws={'color': 'blue', 'linestyle': '--', 'linewidth': 2}, label='Predicted')

plt.title("GT vs. Preds (Prices)")
plt.xlabel("Model Prediction")
plt.ylabel("Ground Truth")
plt.legend()
plt.grid(True)
plt.axis('equal')   # x축과 y축의 스케일을 동일하게 설정합니다.
plt.show()
