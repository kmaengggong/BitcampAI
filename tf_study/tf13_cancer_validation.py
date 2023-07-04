import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score

# 1 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape)  # (569, 30)
# print(y.shape)  # (569,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True
)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(68, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation="sigmoid"))
# 이진분류에서는 마지막 아웃풋 레이어에 꼭 sigmoid 사용해야 됨

# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=32)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
# metrics에 mse, accuracy를 넣었기 때문에, evaluate 시 3가지 값이 나옴
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)  # 1에 가까울수록 좋음

y_predict = model.predict(x_test)
# print(y_predict)
#y_predict2 = []
#for i in range(0, len(y_predict)):
#    y_predict2.append(1 if y_predict[i] > 0.5 else 0)

#y_predict2 = np.where(y_predict > 0.5, 1, 0)
y_predict2 = np.round(y_predict)

#print(y_predict2)
accuracy_score = accuracy_score(y_test, y_predict2)
print("accuracy_score:", accuracy_score)

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
plt.plot(hist.history["loss"], marker=".", c="red", label="loss")
plt.plot(hist.history["val_loss"], marker=".", c="blue", label="val_loss")
plt.title("loss & val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend()
plt.show()
