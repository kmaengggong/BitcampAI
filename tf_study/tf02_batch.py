# 1. 데이터 입력
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

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
## batch_size가 작음
#   모델이 더 자주 업데이트 됨
#   pros  
#       노이즈가 많은 경향이 있음
#   cons
#       더 많은 학습 시간이 필요
## batch_size가 큼
#   학습 데이터의 분포와 유사해짐
#   pros
#       노이즈가 적음
#       빠른 학습 시간
#   cons
#       과적합이 일어날 수 있음(비슷하면 좋은데, 다른 값이 나오면 예측률 낮아짐)
## 일반적으로 32, 64, 128 등 2의 거듭제곱 수를 사용함

model.compile(loss="mae", optimizer="adam")
model.fit(x, y, epochs=1000, batch_size=6)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss:", loss)

result = model.predict([-1])
print("-1의 예측값:", result)