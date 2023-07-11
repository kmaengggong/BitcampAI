import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True)

# 2. 모델 구성
model = RandomForestClassifier(max_depth=6, n_jobs=1)
# model = 

# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 파라미터:", model.best_params_)
print("최적의 매개변수:", model.best_estimator_)
print("best_score:", model.best_score_)
print("model_score:", model.score(x_test, y_test))
print("time:", end_time-start_time)