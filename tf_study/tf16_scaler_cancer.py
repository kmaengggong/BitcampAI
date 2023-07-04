import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

# 1 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape)  # (569, 30)
# print(y.shape)  # (569,)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True
)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(68, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation="sigmoid"))

# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])
model.fit(x_train, y_train, epochs=500, batch_size=32)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)

# None
# loss: 0.2891269624233246
# mse: 0.07099523395299911
# accuracy: 0.9181286692619324

# StandardScaler
# loss: 3.681788682937622
# mse: 0.0530087947845459
# accuracy: 0.9415204524993896

# MinMaxScaler
# loss: 2.624437093734741
# mse: 0.06553371995687485
# accuracy: 0.9239766001701355

# RobustScaler
# loss: 2.884774684906006
# mse: 0.05624194070696831
# accuracy: 0.9415204524993896

# MaxAbsScaler
# loss: 0.8558457493782043
# mse: 0.06382063031196594
# accuracy: 0.9298245906829834