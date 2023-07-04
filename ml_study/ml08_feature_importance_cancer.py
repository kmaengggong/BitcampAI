import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# scaler = RobustScaler()
# x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 7
random_state = 72
kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

# 2. 모델 구성
# model = SVC()
# model = RandomForestClassifier()
model = DecisionTreeClassifier()

# 3. 훈련, 평가
model.fit(x_train, y_train)

score = cross_val_score(model, x, y, cv=kfold)

print("accuracy:", score)
print("cvs:", round(np.mean(score), 4))

##### FEATURE IMPORTANCE #####
import matplotlib.pyplot as plt

n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_, align="center")
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title("Cancer Feature Importances")
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.ylim(-1, n_features)
plt.show()

# 4. 결과
# SVC
# accuracy: [0.97560976 0.97560976 0.96296296 0.98765432 0.98765432 0.97530864 0.96296296]
# cvs: 0.9754

# RFC
# accuracy: [0.95121951 0.95121951 0.96296296 0.98765432 0.96296296 0.95061728 0.98765432]
# cvs: 0.9649