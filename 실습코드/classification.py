import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정
# 그래프에 한글을 표시하기 위한 설정. 시스템 환경(OS)에 따라 폰트 이름이 다를 수 있음
try:
    # Windows는 'Malgun Gothic', macOS는 'AppleGothic'
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        print("[경고] 한글 폰트를 설정하기 못했습니다. 그래프의 한글이 깨질 수 있습니다.")

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# (5) 다중 분류: 로지스틱 회귀 vs. kNN 분류
print("(5) 다중 분류: 로지스틱 회귀 vs. kNN 분류 시작...")

# Digits 데이터셋(손글씨 숫자 이미지, 0~9까지 10개 클래스)을 사용합니다.
digits = load_digits()
X = digits.data
y = digits.target

# 데이터 분할 (학습 70%, 테스트 30%)
# stratify = y: 클래스 비율을 유지하며 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77, stratify=y)

# 로지스틱 회귀 (Logistic Regression)
# multi_class='multinomial': 다중 분류 설정. solver='lbfgs': 효율적인 최적화 알고리즘
log_reg_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=77)
log_reg_clf.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg_clf.predict(X_test_scaled)

# 모델 평가 함수 정의
def evaluate_model(y_true, y_pred, model_name):
    # 모델 평가 지표(정확도, 정밀도, 재현율, F1 점수) 계산 및 출력
    print(f"\n[{model_name} 평가 결과]")
    # 정확도 (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
    # 정밀도, 재현율, F1 점수 ('macro' 평균은 모든 클래스를 동등한 비중으로 고려)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"정확도 (Accuracy): {accuracy:.4f}")
    print(f"정밀도 (Precision): {precision:.4f}")
    print(f"재현율 (Recall): {recall:.4f}")
    print(f"F1 점수 (F1-Score): {f1:.4f}")

    # 혼동 행렬 (Confusion Matrix)
    return confusion_matrix(y_true, y_pred)

# 각 모델 평가
cm_log_reg = evaluate_model(y_test, y_pred_log_reg, "로지스틱 회귀")

# 시각화: 혼동 행렬 비교
plt.figure(figsize=(6, 6))

# 로지스틱 회귀 혼동 행렬
# annot=True: 셀 안에 숫자를 표시. fmt='d': 정수형으로 표시.
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title("로지스틱 회귀 혼동 행렬")
plt.xlabel("예측된 레이블 (Predicted)")
plt.ylabel("실제 레이블 (True)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
