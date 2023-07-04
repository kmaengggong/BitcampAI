import time
import matplotlib.pyplot as plt

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.callbacks import EarlyStopping

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

# plt.imshow(x_train[0])
# plt.show()

# SCALING: 0~255 -> 0~1
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train/255
x_test = x_test/255

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32, 32, 3)))
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
model.add(Dense(256, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(100, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics="accuracy")

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train, validation_split=0.2, callbacks=early_stopping, epochs=100)
end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("accuracy:", acc)
print("time:", end_time-start_time)

# 5. 결과