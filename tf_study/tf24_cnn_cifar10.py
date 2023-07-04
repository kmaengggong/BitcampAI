import time
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

# 시각화
# plt.imshow(x_train[0])
# plt.show()

# RESHAPE - 이미 , , , 3이므로 안함
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

# print(x_train.shape, x_test.shape)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train/255
x_test = x_test/255

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(2, 2))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation="relu"))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))  # 데이터 수를 줄이는 거
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=7,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=early_stopping)
end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("accuracy:", acc)
print("time:", end_time-start_time)

# 5. 결과 - padding, dropout 위치랑 수치, pooling
# loss: 1.0875160694122314
# accuracy: 0.6245999932289124
# time: 1189.2209346294403