import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from keras.callbacks import EarlyStopping

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
    x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True
)

# print(x_train.shape)  # 105, 4
# print(x_test.shape)  # 45, 4
# print(y_train.shape)
# print(y_test.shape)

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(105, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])

earlyStopping = EarlyStopping(
    monitor="val_loss",
    patience=100,
    mode="min",
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=earlyStopping)
end_time = time.time()

print("running_time:", end_time-start_time)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)

y_predict = model.predict(x_test)

# None
# loss: 0.02389426715672016
# mse: 0.002600201405584812
# accuracy: 1.0

# StandardScaler
# loss: 0.022482892498373985
# mse: 0.0038087978027760983
# accuracy: 1.0

# MinMaxScaler
# loss: 0.03820376843214035
# mse: 0.0063987718895077705
# accuracy: 1.0

# RobustScaler

# MaxAbsScaler

# StandardScaler + EalryStopping
# loss: 0.011397435329854488
# mse: 0.0004640388360712677
# accuracy: 1.0