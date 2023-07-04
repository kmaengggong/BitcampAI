from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape)  # 178, 13
# print(y.shape)  # 178
# print(datasets.feature_names)
# print(datasets.DESCR)

# one-hot encoding
y = to_categorical(y)
# print(y.shape)  # 178, 3

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, random_state=1711, shuffle=True
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32)
end_time = time.time()

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)
print("running_time:", end_time-start_time)

# 5. 결과
# None
# loss: 0.22874963283538818
# mse: 0.04888613894581795
# accuracy: 0.8888888955116272
# running_time: 6.603365659713745

# StandardScaler
# loss: 0.33802998065948486
# mse: 0.023696133866906166
# accuracy: 0.9629629850387573
# running_time: 6.163255453109741