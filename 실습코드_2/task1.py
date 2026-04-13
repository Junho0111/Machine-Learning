# 실습 과제 1: Support vector classifier (SVC) vs. Logistic regression
# 실습 목표: 유방암 진단 데이터셋을 사용하여 SVC와 Logistic regression 모델을 학습하고, 성능을 비교 분석
# 데이터 셋: scikit-learn의 'load_breast_cancer' 데이터
# 평가 지표: 정확도(accuracy), 정밀도(precision), 재현율(recall), F1-score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# [제공 함수] 모델 평가 결과 시각화 함수
def visualize_classification_performance(results_df):
    """
    분류 모델들의 성능 지표를 비교하는 막대 그래프를 생성합니다.
    Args:
        results_df (pd.DataFrame): 모델 이름, 평가지표, 성능 점수를 포함하는 데이터 프레임
                                    (컬럼: ['Model', 'Metric', 'Score'])
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(data=results_df, x='Metric', y='Score', hue='Model')
    plt.title("SVC vs. Logistic regression", fontsize=16)
    plt.xlabel('Evaluation metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0.9, 1.0)  # 점수 범위를 조정하여 차이를 명확하게 보여줌
    plt.legend(title='Model', loc='lower right')
    plt.show()

# [실습] 아래의 코드를 완성하세요.
def main():
    # Step 1. 데이터 로드
    # Hint: load_breast_cancer() 함수 사용
    


    # Step 2. 데이터 분할
    # Hint: train_test_split() 함수 사용
    # 조건: test_size=0.3, random_state=77, stratify=y



    # Step 3. 데이터 스케일링
    # Hint 1: StandardScaler 사용
    # Hint 2: train 데이터에 대해서만 fit()



    # Step 4. 모델 학습
    # Hint: SVC와 LogisticRegression 모델 객체를 각각 생성하고 학습(fit)
    # 조건 1: 두 모델 모두 random_state=77로 설정
    # 조건 2: SVC의 kernel은 'linear'로 설정
    # 조건 3: Logistic Regression의 penalty는 None으로 설정

    # 4-1. Support Vector Classifier (SVC) 학습


    # 4-2. Logistic Regression 학습



    # Step 5. 모델 평가
    # Hint 1: 학습된 모델을 사용하여 test 데이터에 대한 예측(predict)을 수행
    # Hint 2: 이후 예측 결과와 실제 정답을 사용하여 평가 지표를 계산

    # 5-1. SVC 모델 평가



    # 5-2. Logistic regression 모델 평가




    # Step 6. 결과 종합 및 시각화
    # Hint 1: 평가 결과를 Pandas DataFrame으로 만들고, 제공된 시각화 함수를 호출
    results = []
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    svc_scores = [accuracy_svc, precision_svc, recall_svc, f1_svc]  # Hint 2
    lr_scores = [accuracy_lr, precision_lr, recall_lr, f1_lr]       # Hint 3

    for i, metric in enumerate(metrics):
        results.append({'Model': 'SVC', 'Metric': metric, 'Score': svc_scores[i]})
        results.append({'Model': 'Logistic Regression', 'Metric': metric, 'Score': lr_scores[i]})

    results_df = pd.DataFrame(results)
    
    print("성능 비교 결과:\n", results_df)

    # 제공된 시각화 함수 호출
    visualize_classification_performance(results_df)

if __name__ == '__main__':
    main()
