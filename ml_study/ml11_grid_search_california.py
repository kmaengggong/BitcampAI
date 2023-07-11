import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=72, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
random_state = 72
kfold = KFold(n_splits=n_split, random_state=random_state, shuffle=True)

# param = [
#     {'n_estimators':[100, 500], 'max_depth':[1,6,]}
# ]

# 2. 모델 구성
rf_model = RandomForestRegressor()
model = GridSearchCV(rf_model, param, cv=kfold, verbose=1, repeat=True, n_jobs=-1)

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()