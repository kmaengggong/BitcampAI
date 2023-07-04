import time
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)

# 시각화
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()

# RESHAPE
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape, x_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4), padding="same", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(16, (2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# 3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
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

# 5. 결과
# loss: 0.3588612973690033
# accuracy: 0.8733999729156494
# time: 100.89974999427795