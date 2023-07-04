import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
#print(x.shape)  # (20640, 8)
#print(y.shape)  # (20640,)
#print(datasets.feature_names)
#print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True
)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# Scaler(정규화) 적용
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = load_model("./_save/tf18_save_model_california.h5")

# 3. 컴파일, 훈련

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

# 4-1. R2 Score 결정계수
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2:", r2)