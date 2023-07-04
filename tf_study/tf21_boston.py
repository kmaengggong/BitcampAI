import time, datetime
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
filepath = './data/boston/'
x_train = pd.read_csv(filepath + 'train-data.csv', index_col=0)
y_train = pd.read_csv(filepath + 'train-target.csv', index_col=0)
x_test = pd.read_csv(filepath + 'test-data.csv', index_col=0)
y_test = pd.read_csv(filepath + 'test-target.csv', index_col=0)

print(x_train.shape, x_test.shape)  # (333, 11) (173, 11)
print(y_train.shape, y_test.shape)  # (333, 0) (173, 0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=11))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics="mse")

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=50,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32)
end_time = time.time()

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("loss:", loss)
print("mse:", mse)
print("r2:", r2)
print("running_time:", end_time-start_time)
