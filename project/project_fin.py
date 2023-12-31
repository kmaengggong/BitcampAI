

import time, datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DL
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. 데이터
# data_path = "../data/credit_card_prediction/"  # 리눅스 환경
data_path = "./data/credit_card_prediction/"
datasets = pd.read_csv(data_path + "train.csv")

# 1.1 NaN값 처리 - 최빈값으로 처리
datasets = datasets.fillna(datasets["Credit_Product"].value_counts().idxmax())

string_columns = ["ID", "Gender", "Region_Code", "Occupation", "Channel_Code", "Credit_Product", "Is_Active"]
label_encoder = LabelEncoder()
for i in range(len(string_columns)):
    datasets[string_columns[i]] = label_encoder.fit_transform(datasets[string_columns[i]])

x = datasets.drop(columns="Is_Lead")
y = datasets.Is_Lead
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True)

# 1-1. 데이터 확인
# Columns
# ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active', 'Is_Lead']
# Target = Is_Lead, value=[0, 1]
# print(datasets.shape)  # (245725, 11)

# 1-2. 스케일링
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 1-3. 데이터 시각화
# sns.set(font_scale=1.2)
# sns.set(rc = {"figure.figsize":(12, 9)})
# sns.heatmap(data=datasets.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 1-4. Outlier
# outlers = np.array([
#         -50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#         11, 12, 13, 14, 15, 16, 17, 18, 50, 100
# ])

# def outliers(data_out):
#     q1, q2, q3 = np.percentile(data_out, [25, 50, 75])
#     print('1사분위 : ', q1)
#     print('2사분위 : ', q2)
#     print('3사분위 : ', q3)
    
#     iqr =  q3 - q1
#     print(iqr)

#     lower_bound = q1 - (iqr * 1.5)
#     upper_bound = q3 + (iqr * 1.5)
#     print('lower_bound : ', lower_bound)
#     print('upper_bound : ', upper_bound)

#     return np.where((data_out > upper_bound) | (data_out < lower_bound))

# outlers_loc = outliers(outlers)
# print('이상치의 위치 : ', outlers_loc)

# plt.boxplot(outlers_loc)
# plt.show()

# ML
# n_splits = 7
# random_state = 72
# kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)


# 2. 모델 구성
# DL
model = Sequential()
model.add(Dense(128, input_dim=10))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(1, activation="sigmoid"))

# ML
# model = SVC()
# model = RandomForestClassifier()
# model = DecisionTreeClassifier()


# 3. 컴파일, 훈련
# DL
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[early_stopping])
end_time = time.time()

# ML
# start_time = time.time()
# model.fit(x_train, y_train)
# end_time = time.time()

# 4. 평가, 예측
# DL
loss, acc = model.evaluate(x_test, y_test)
print("loss:", loss)
print("acc:", acc)
print("time:", end_time-start_time)

# ML
# score = cross_val_score(model, x, y, cv=kfold)
# print("acc:", round(np.mean(score), 4))
# print("time:", end_time-start_time)

# 4-1. Feature Importances 시각화(ML)
# n_features = x.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align="center")
# plt.yticks(np.arange(n_features), x.columns)
# plt.title("Credit Card Feature Importances")
# plt.ylabel("Feature")
# plt.xlabel("Importance")
# plt.ylim(-1, n_features)
# plt.show()


# 5. 결과
# DL / StandardScaler / 128 64 32 64 128 1 / epochs=100, batch_size=128
# loss: 0.5007683038711548
# acc: 0.7551124095916748
# time: 65.36959648132324

# ML / StandardScaler / DecisionTreeClassifier
# acc: 0.71
# time: 0.8604931831359863