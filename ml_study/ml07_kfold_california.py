import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True)

scaler = StandardScaler()
x  = scaler.fit_transform(x)

kfold = KFold(n_splits=5, random_state=72, shuffle=True)

# 2. 모델 구성
# model = SVR()
model = RandomForestRegressor()

# 3. 훈련, 평가
score = cross_val_score(model, x, y, cv=kfold)

print("r2:", score)
print("cvs:", round(np.mean(score), 4))

# 4. 결과
# SVR
# r2: [0.74977437 0.72822127 0.73631372 0.75289926 0.72902466]
# cv: 0.7392

# RFR
# r2: [0.81269629 0.80323997 0.80941615 0.81381337 0.80522904]
# cvs: 0.8089