from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime
import matplotlib.pyplot as plt

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

earlyStopping = EarlyStopping(
    monitor="val_loss",
    mode="mid",
    patience=50,
    verbose=1,
    restore_best_weights=True
)

filepath = './_mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

mcp = ModelCheckpoint(
    monitor="val_loss",
    mode="auto",
    verbose=1,
    save_best_only=True,
    # filepath="./_mcp/tf20_california.hdf5"
    filepath="".join([filepath, 'tf20_', date, '_', filename])
)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[earlyStopping, mcp])
end_time = time.time()

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("loss:", loss)
print("mse:", mse)
print("r2:", r2)
print("running_time:", end_time-start_time)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history["loss"], marker=".", c="red", label="loss")
# plt.plot(hist.history["val_loss"], marker=".", c="blue", label="val_loss")
# plt.title("loss & val_loss")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.legend()
# plt.show()