# [실습]
# 1. iris 데이터에 CatboostClassfier 적용하여 결과 도출
# 2. cancer, california 데이터에 boosting 3대장(xgboost, lgbm, catboost)를 적용하여 코드 완성
# 3. boosting 3대장 성능 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. 데이터
datasets = load_iris()
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
# acc: [1.         0.93333333 0.93333333 0.96666667 0.96666667]
# cvs: 0.96

# lgbm
# acc: [1.         0.93333333 0.9        0.93333333 1.        ]
# cvs: 0.9533

# xgboost
# acc: [1.         0.93333333 0.9        0.9        1.        ]
# cvs: 0.9467