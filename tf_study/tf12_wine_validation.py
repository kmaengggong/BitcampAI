from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import time

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape)  # 178, 13
print(y.shape)  # 178
# print(datasets.feature_names)
# print(datasets.DESCR)

# one-hot encoding
#y = to_categorical(y)
# print(y.shape)  # 178, 3

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=1711, shuffle=True
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation="softmax"))
# relu: -> 
# softmax: 출력값을 0~1 사이 값으로 -> 다중 분류
# sigmoid: 출력값을 0 or 1로 -> 이진 분류

# 3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])

start_time = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=32)
end_time = time.time()

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)
print("running_time:", end_time-start_time)