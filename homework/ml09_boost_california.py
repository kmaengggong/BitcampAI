# [실습]
# 1. iris 데이터에 CatboostClassfier 적용하여 결과 도출
# 2. cancer, california 데이터에 boosting 3대장(xgboost, lgbm, catboost)를 적용하여 코드 완성
# 3. boosting 3대장 성능 비교

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

scaler = RobustScaler()
x  = scaler.fit_transform(x)

kfold = KFold(n_splits=5, random_state=72, shuffle=True)

# 2. 모델 구성
# model = CatBoostRegressor()
# model = LGBMRegressor()
model = XGBRegressor()


# 3. 훈련, 평가
model.fit(x, y)

score = cross_val_score(model, x, y, cv=kfold)

print("r2:", score)
print("cvs:", round(np.mean(score), 4))

# 4. 결과
# Catboost
# r2: [0.85727136 0.85012636 0.84547068 0.85497032 0.84659664]
# cvs: 0.8509

# lgbm
# r2: [0.84468356 0.83740061 0.83147745 0.83993431 0.83372604]
# cvs: 0.8374

# xgboost
# r2: [0.83353791 0.83414884 0.82718217 0.84170229 0.82721194]
# cvs: 0.8328