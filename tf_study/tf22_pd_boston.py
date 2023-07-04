import datetime, time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

# 1. 데이터
csvpath = './data/boston/'
datasets = pd.read_csv(csvpath + 'Boston_house.csv')
# print(datasets)  # (506, 14)
# print(datasets.columns)

# x = datasets.drop(columns="Target")
x = datasets.drop(['Target'], axis=1)
y = datasets.Target
# print(x.shape)  # (506, 13)
# print(y.shape)  # (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2, random_state=72, shuffle=True)

# print(x_train.shape, x_test.shape)  # (303, 13) (102, 13)
# print(y_train.shape, y_test.shape)  # (303,) (102,)

# scaler = StandardScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics="mse")

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=50,
    verbose=1,
    restore_best_weights=True
)

date = datetime.datetime.now().strftime("%m%d_%H%M")
savepath = "./_mcp/"
savename = "_{epoch:04d}-{val_loss:.4f}.hdf5"

mcp = ModelCheckpoint(
    monitor="val_loss",
    mode="auto",
    verbose=1,
    save_best_only=True,
    filepath="".join([savepath, "tf22_boston_", date, savename])
)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[early_stopping, mcp], verbose=1)
end_time = time.time()

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("loss:", loss)
print("mse:", mse)
print("r2:", r2)
print("running_time:", end_time-start_time)

sns.set(font_scale=1.2)
sns.set(rc={'figure.figsize':(9,6)})
sns.heatmap(
    data=datasets.corr(),  # .corr(): 상관관계(Coreleation)
    square=True,  # view를 정사각형으로
    annot=True,  # 각 cell값 표기(annotation)
    #color=True
)
plt.show()

# 5. 결과값
# 1. Flat
# D(128, 128, 128, 128) SS, ES
# loss: 25.357940673828125
# mse: 25.357940673828125
# r2: 0.689602029070618
# running_time: 5.720287561416626

# 2. Scaler Change
# D(128, 128, 128, 128) RS, ES
# loss: 24.92822265625
# mse: 24.92822265625
# r2: 0.6948620743494267
# running_time: 4.811566352844238

# 3. Increasing
# D(32, 64, 128, 256, 512) RS, ES
# loss: 28.881460189819336
# mse: 28.881460189819336
# r2: 0.6464718372795346
# running_time: 4.410711050033569

# 4. Bottleneck
# D(128, 64, 32, 64, 128) RS, ES
# loss: 24.594058990478516
# mse: 24.594058990478516
# r2: 0.6989524572958652
# running_time: 7.052416563034058

# 5. Batchsize to 64
# D(128, 64, 32, 64, 128) RS, ES
# loss: 24.836742401123047
# mse: 24.836742401123047
# r2: 0.6959818757318291
# running_time: 6.634795188903809

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 6. Increase Hidden Layers
# D(128, 64, 32, 16, 32, 64, 128) RS, ES
# loss: 23.657041549682617
# mse: 23.657041549682617
# r2: 0.7104221547942133
# running_time: 5.747037887573242
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 7. Increase more Hidden Layers
# D(128, 64, 32, 16, 32, 64, 128) RS, ES
# loss: 25.004560470581055
# mse: 25.004560470581055
# r2: 0.6939276371692537
# running_time: 8.541464805603027