# 1. 데이터 입력
import numpy as np
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
# mse = 평균 제곱 오차
# mae = 평균 절대값 오차
model.compile(loss="mae", optimizer="adam")
# mse: loss: 0.37999993562698364
# mae: loss: 0.4146619439125061
model.fit(x, y, epochs=1000)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss:", loss)

result = model.predict([6])
print("6의 예측값:", result)
# mse: 6의 예측값: [[5.699998]]
# mae: 6의 예측값: [[5.817876]]