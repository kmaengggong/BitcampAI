# [실습]
# 1. iris 데이터에 CatboostClassfier 적용하여 결과 도출
# 2. cancer, california 데이터에 boosting 3대장(xgboost, lgbm, catboost)를 적용하여 코드 완성
# 3. boosting 3대장 성능 비교

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

scaler = RobustScaler()
x  = scaler.fit_transform(x)

kfold = StratifiedKFold(n_splits=5, random_state=72, shuffle=True)

# 2. 모델 구성
# model = CatBoostClassifier(iterations=2, depth=2, learning_rate=1)
# model = LGBMClassifier()
model = XGBClassifier()


# 3. 훈련, 평가
model.fit(x, y)

score = cross_val_score(model, x, y, cv=kfold)

print("acc:", score)
print("cvs:", round(np.mean(score), 4))

# 4. 결과
# Catboost
# acc: [0.89473684 0.90350877 0.93859649 0.89473684 0.90265487]
# cvs: 0.9068

# lgbm
# acc: [0.93859649 0.94736842 0.95614035 0.97368421 0.97345133]
# cvs: 0.9578

# xgboost
# acc: [0.92982456 0.96491228 0.96491228 0.98245614 0.98230088]
# cvs: 0.9649