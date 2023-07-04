import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1, 21))
y = np.array(range(1, 21))

# train:test = 70:30
# train이 너무 많으면 과적합
x_train = np.array(range(1, 15))
y_train = np.array(range(1, 15))

x_test = np.array(range(15, 21))
y_test = np.array(range(15, 21))

# 2. 모델 구성
model = Sequential()
model.add(Dense(14, input_dim=1))  # train 데이터 갯수 이하로
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(75))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

result = model.predict([21])
print("21의 예측값:", result)

# D(10, 25, 50, 75, 100)
# loss: 2.4253192770079535e-12
# 21의 예측값: [[21.]]