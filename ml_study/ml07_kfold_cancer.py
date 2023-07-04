import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

scaler = RobustScaler()
x = scaler.fit_transform(x)

n_splits = 7
random_state = 72
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

# 2. 모델 구성
# model = SVC()
model = RandomForestClassifier()

# 3. 훈련, 평가
score = cross_val_score(model, x, y, cv=kfold)

print("accuracy:", score)
print("cvs:", round(np.mean(score), 4))

# 4. 결과
# SVC
# accuracy: [0.97560976 0.97560976 0.96296296 0.98765432 0.98765432 0.97530864 0.96296296]
# cvs: 0.9754

# RFC
# accuracy: [0.95121951 0.95121951 0.96296296 0.98765432 0.96296296 0.95061728 0.98765432]
# cvs: 0.9649