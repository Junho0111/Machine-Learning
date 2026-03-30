
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
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

# 실습과제 1: Wine 데이터셋 분류 및 스케일링 효과 확인
print("Task (1): Wine 데이터셋 분류 시작...")

# 1. 데이터 로드 (Wine 데이터셋: 와인 화학 성분을 바탕으로 품종(3종) 분류) - load_wine() 함수 X는 data, y는 target
wine = load_wine()



# 2. 데이터 분할 (학습 75%, 테스트 25%) + stratify 적용




# 3. 데이터 전처리 (표준 스케일링)
# 학습 데이터에 fit하고 transform
# 테스트 데이터는 학습 데이터의 기준으로 transform만 수행






# 4. 로지스틱 회귀 (max_iter=1000, random_state=77)





# 5. 모델 평가 (스케일링 적용 O)
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

# 평가 함수를 사용하여 모델 평가 및 혼동 행렬 생성
print("\n--- 실습과제 결과 ---")
# cm_lr_wine = ...




# 시각화: 혼동 행렬
plt.figure(figsize=(6, 6))
sns.heatmap(cm_lr_wine, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title("로지스틱 회귀 (스케일링 O)")
plt.xlabel("예측된 레이블")
plt.ylabel("실제 레이블")

plt.tight_layout()
plt.show()
