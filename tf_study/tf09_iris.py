import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(x.shape)  # 150, 4
# print(y.shape)  # 150,
# print(datasets.feature_names)
# print(datasets.DESCR)

# one-hot encoding
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True
)

print(x_train.shape)  # 105, 4
print(x_test.shape)  # 45, 4
print(y_train.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(105, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])
# 회귀분석은 mse, r2 score
# 분류분석은 mse, accuracy score

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32)
end_time = time.time()

print("running_time:", end_time-start_time)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)

y_predict = model.predict(x_test)