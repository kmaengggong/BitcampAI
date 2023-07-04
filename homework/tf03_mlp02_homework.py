import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x_transpose = x.transpose()

# 모델구성부터 평가예측까지 완성하시오
# 예측 [[10, 1.6, 1]]

# 2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(75))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_transpose, y, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_transpose, y)
print("loss:", loss)

result = model.predict([[10, 1.6, 1]])
print("[[10, 1.6, 1]]의 예측값:", result)  # 20

# Dense(100)
# loss: 0.00010280384594807401
# result: [[19.997375]]
# Dense(10), Dense(50), Dense(100)
# loss: 2.1118878066772595e-05
# result: [[19.998371]]
# Dense(10), Dense(25), Dense(50), Dense(75), Dense(100)
# loss: 3.1104719494523536e-11
# result: [[20.]]