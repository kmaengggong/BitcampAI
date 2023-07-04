# 1. 데이터 입력
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(10))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=1000)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss:", loss)

result = model.predict([4])
print("4의 예측값:", result)

result = model.predict([100])
print("100의 예측값:", result)