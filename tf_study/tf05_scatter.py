import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(1, 21))
y = np.array(range(1, 21))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,  # 데이터
    train_size=0.7,  # train set 70%
    test_size=0.3,  # test set 30%, 둘 중 하나만 적어도 됨
    random_state=123,  # 데이터를 난수값에 의해 추출한다는 의미
    shuffle=True
)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1000, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x)

import matplotlib.pyplot as plt
plt.scatter(x, y, color='black')  # 산정도 그리기
plt.plot(x, y_predict, color='red')
plt.show()