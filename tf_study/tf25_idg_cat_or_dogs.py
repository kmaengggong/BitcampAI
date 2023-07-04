import time, datetime
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 스케일 조정
    horizontal_flip=True,  # 수평으로 반전
    vertical_flip=True,  # 수직으로 반전
    width_shift_range=0.1,  # 수평 이동 범위
    height_shift_range=0.1,  # 수직 이동 범위
    rotation_range=5,  # 회전 범위
    zoom_range=1.2,  # 확대 범위
    shear_range=0.5,  # 기울이기 범위
    fill_mode="nearest"  # 채우기 범위
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './data/cat_dog/training_set/',
    target_size=(150, 150),
    batch_size=8005,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True
)
# print(xy_train)  # 8005, 2 classes

xy_test = test_datagen.flow_from_directory(
    './data/cat_dog/test_set/',
    target_size=(150, 150),
    batch_size=2023,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True
)
# print(xy_test)  # 2023, 2 classes

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(150, 150, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.1))
model.add(Conv2D(32, (2, 2), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 3. 컴파일, 훈련
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(xy_train[0][0], xy_train[0][1], epochs=100, batch_size=128, validation_split=0.2, callbacks=early_stopping)
end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(xy_test[0][0], xy_test[0][1])
print("loss:", loss)
print("acc:", acc)
print("time:", end_time-start_time)