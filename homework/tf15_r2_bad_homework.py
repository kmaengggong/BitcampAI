# [실습]
# 1. r2 score를 음수가 아닌 0.5 이하로
# 2. 데이터는 그대로
# 3. 레이어는 인풋, 아웃풋 포함 7개 이상
# 4. batch_Size=1
# 5. 히든 레이어의 노드(뉴런) 갯수는 10개 이상 100개 이하로
# 6. train_size=0.7
# 7. epochs=100 이상

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
x = np.array(range(1, 21))
y = np.array(range(1, 21))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=21, shuffle=False
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1111, input_dim=1))

model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))

model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))

model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))

model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))
model.add(Dense(21))

model.add(Dense(21))

model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mae")
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
r2_score = r2_score(y_test, y_predict)
print("r2:", r2_score)

# min_r2: 0.3