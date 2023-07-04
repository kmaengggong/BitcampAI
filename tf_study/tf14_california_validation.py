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
# 1-1. 데이터 확인
#print(x.shape)  # (20640, 8)
#print(y.shape)  # (20640,)
#print(datasets.feature_names)
#print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=731, shuffle=True
)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=8))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(2))
model.add(Dense(32))
model.add(Dense(128))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics="mse")
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=128)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss:", loss)
print("mse:", mse)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history["loss"], marker=".", c="red", label="loss")
plt.plot(hist.history["val_loss"], marker=".", c="blue", label="val_loss")
plt.title("loss & val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend()
plt.show()