import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=72, shuffle=True)

# 2. 모델 구성
rf_model = RandomForestRegressor()
model = BaggingRegressor(rf_model, n_estimators=100, n_jobs=-1, random_state=72)

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

# 4. 평가
result = model.score(x_test, y_test)

print("r2:", result)
print("time:", end_time-start_time)