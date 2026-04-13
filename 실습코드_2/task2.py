import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# [제공 함수] 모델 평가 결과 시각화 함수
def visualize_regression_performance(y_test, y_pred_svr, y_pred_lr):
    """
    회귀 모델들의 성능을 실제값-예측값 산점도와 평가지표 막대그래프로 시각화합니다.
    Args:
        y_test (pd.Series): 테스트 데이터의 실제 타겟값
        y_pred_svr (np.array): SVR 모델의 예측값
        y_pred_lr (np.array): Linear regression 모델의 예측값
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # SVR plot
    sns.scatterplot(x=y_test, y=y_pred_svr, ax=axes[0], alpha=0.6)
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2)
    axes[0].set_title("SVR: Real vs. Predicted", fontsize=15)
    axes[0].set_xlabel('Real price', fontsize=12)
    axes[0].set_ylabel('Predicted price', fontsize=12)
    
    # Linear regression plot
    sns.scatterplot(x=y_test, y=y_pred_lr, ax=axes[1], alpha=0.6)
    axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', lw=2)
    axes[1].set_title("Linear regression: Real vs. Predicted", fontsize=15)
    axes[1].set_xlabel('Real price', fontsize=12)
    axes[1].set_ylabel('Predicted price', fontsize=12)

    plt.tight_layout()
    plt.show()

    # 평가 지표 계산
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    metrics_df = pd.DataFrame({
        'Model': ['SVR', 'Linear Regression', 'SVR', 'Linear Regression'],
        'Metric': ['MSE', 'MSE', 'R2 Score', 'R2 Score'],
        'Score': [mse_svr, mse_lr, r2_svr, r2_lr]
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=metrics_df[metrics_df['Metric'] == 'MSE'], x='Model', y='Score', ax=axes[0])
    axes[0].set_title("MSE (Mean Squared Error) Comparison", fontsize=15)
    axes[0].set_ylabel('MSE value', fontsize=12)

    sns.barplot(data=metrics_df[metrics_df['Metric'] == 'R2 Score'], x='Model', y='Score', ax=axes[1])
    axes[1].set_title("R2 Score Comparison", fontsize=15)
    axes[1].set_ylabel('R2 Score value', fontsize=12)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

# [실습] 아래의 코드를 완성하세요.
def main():
    # Step 1. 데이터 로드
    # Hint: fetch_california_housing() 함수 사용
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    # Step 2. 데이터 분할
    # Hint: train_test_split() 함수 사용
    # 조건: test_size=0.3, random_state=77
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)

    # Step 3. 데이터 스케일링
    # Hint 1: StandardScaler 사용
    # Hint 2: train 데이터에 대해서만 fit()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # 학습데이터 fit_transform
    X_test_scaled = scaler.transform(X_test) # 테스트 데이터 학습데이터 기준으로 fransform

    # Step 4. 모델 학습
    # Hint: SVR와 LinearRegression 모델 객체를 각각 생성하고 학습(fit)
    # 조건: SVR의 kernel은 'rbf'로 설정 (gamma='auto')

    # 4-1. Support Vector Regressor (SVR) 학습
    svr = SVR(kernel='rbf', gamma='auto')
    svr.fit(X_train_scaled, y_train)

    # 4-2. Linear Regression 학습
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Step 5. 모델 평가
    # Hint 1: 학습된 모델을 사용하여 test 데이터에 대한 예측(predict)을 수행
    # Hint 2: 이후 예측 결과와 실제 정답을 사용하여 평가 지표를 계산

    # 5-1. SVR 모델 평가
    y_pred_svr = svr.predict(X_test_scaled)             # 예측 값
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    print("\n[SVR 모델 평가 결과]")
    print(f"  - Mean Squared Error (MSE): {mse_svr:.4f}")
    print(f"  - R-squared (R2): {r2_svr:.4f}")

    # 5-2. Linear regression 모델 평가
    y_pred_lr = lr.predict(X_test_scaled)             # 예측 값
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    print("\n[Linear Regression 모델 평가 결과]")
    print(f"  - Mean Squared Error (MSE): {mse_lr:.4f}")
    print(f"  - R-squared (R2): {r2_lr:.4f}")

    # Step 6. 제공된 시각화 함수 호출
    # Hint: visualize_regression_performance()
    visualize_regression_performance(y_test, y_pred_svr, y_pred_lr)

if __name__ == '__main__':
    main()
