import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array(range(1, 21))
y = np.array(range(1, 21))

# train:val:test = 60:20:20
x_train = np.array(range(1, 13))
y_train = np.array(range(1, 13))

x_val = np.array(range(13, 17))
y_val = np.array(range(13, 17))

x_test = np.array(range(17, 21))
y_test = np.array(range(17, 21))

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
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=[x_val, y_val])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

result = model.predict([21])
print("21의 예측값:", result)

# D(10, 25, 50, 75, 100)
# loss: 8.185452315956354e-12
# 21의 예측값: [[21.]]