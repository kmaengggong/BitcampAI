import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([
    range(1, 11),
    [1, 2.1, 3.1, 4.1, 5.1, 6, 7, 8.1, 9.2, 10.5]
])
y = np.array(
    range(11, 21)
)

print(x)
print(y)

print(x.shape)
print(y.shape)

x_transpose = x.transpose()  # == x.T
print(x_transpose.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_transpose, y, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_transpose, y)
print("loss:", loss)

result = model.predict([[10, 10.5]])
print("10의 예측값:", result)