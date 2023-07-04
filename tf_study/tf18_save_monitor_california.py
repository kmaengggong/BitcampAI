import numpy as np
from keras.models import Sequential
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
model = Sequential()
model.add(Dense(128, activation="linear", input_dim=8))  # linear가 deafult
model.add(Dense(128, activation="relu"))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1, activation="linear"))

model.save("./_save/tf18_save_monitor_california.h5")

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")

# Early Stopping 적용
from keras.callbacks import EarlyStopping

ealryStopping = EarlyStopping(
    monitor="val_loss",
    patience=50,
    mode="min",
    verbose=1,
    restore_best_weights=True  # rbw의 default = False. 최적의 weight를 찾기 위해 True로.
)

model.fit(x_train, y_train, validation_split=0.2, callbacks=ealryStopping, epochs=5000, batch_size=128)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

# 4-1. R2 Score 결정계수
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2:", r2)