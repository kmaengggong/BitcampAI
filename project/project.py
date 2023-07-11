import time, datetime
import numpy as np
import tensorflow as tf

# DL
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder

# ML

# 1. 데이터
data_path = "./data/credit_card_prediction/"
datasets = pd.read_csv(data_path + "train.csv")

# NaN값 처리
# train = train.fillna(0)
# train = train.fillna(train.mean())
# train = train.fillna(train.mode())
# train = train.fillna(train.min())
# train = train.fillna(train.max())
# train = train.fillna(method="ffill")
# train = train.fillna(method="bfill")
datasets = datasets.fillna(datasets["Credit_Product"].value_counts().idxmax())

string_columns = ["ID", "Gender", "Region_Code", "Occupation", "Channel_Code", "Credit_Product", "Is_Active"]
label_encoder = LabelEncoder()
for i in range(len(string_columns)):
    datasets[string_columns[i]] = label_encoder.fit_transform(datasets[string_columns[i]])

print(len(datasets.columns))
for i in range(len(datasets.columns)):
    print(datasets[datasets.columns[i]])

x = datasets.drop(columns="Is_Lead")
# print(x)
# x_train = x_train.astype(np.float32)
# x_train = x_train.values
# x_train = tf.convert_to_tensor(x_train)
y = datasets.Is_Lead
# y_train = y_train.astype(np.float32)
# y_train = y_train.values
# y_train = tf.convert_to_tensor(y_train)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True
# )
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True)

# x_test = pd.read_csv(data_path + "test.csv")
# y_test = pd.read_csv(data_path + "sample_submission.csv")
# y_test = y_test.drop(columns="ID")
# y_test = y_test.astype(np.float32)

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True)

# 1-1. 데이터 확인
# print(train.columns)
# Columns
# ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active', 'Is_Lead']
# Target = Is_Lead, value=[0, 1]
# print(train.shape)  # (245725, 11)
# print(test.shape)  # (105312, 10) - Is_Lead가 빠짐
# print(test.columns)
# print(x_train.shape)  # (245725, 10)
# print(y_train.shape)  # (245725,)
# print(x_test.shape)  # (105312, 10)
# print(y_test.shape)  # (105312, 1)

# 1-1. 스케일링
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# DL
model = Sequential()
model.add(Dense(128, input_dim=10))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(1, activation="sigmoid"))

# 3. 컴파일, 훈련
# DL
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128)
end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print("loss:", loss)
print("acc:", acc)
print("time:", end_time-start_time)