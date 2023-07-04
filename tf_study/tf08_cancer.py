import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score

# 1 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape)  # (569, 30)
# print(y.shape)  # (569,)
# print(datasets.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
# 'mean smoothness' 'mean compactness' 'mean concavity'
# 'mean concave points' 'mean symmetry' 'mean fractal dimension'
# 'radius error' 'texture error' 'perimeter error' 'area error'
# 'smoothness error' 'compactness error' 'concavity error'
# 'concave points error' 'symmetry error' 'fractal dimension error'
# 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
# 'worst smoothness' 'worst compactness' 'worst concavity'
# 'worst concave points' 'worst symmetry' 'worst fractal dimension']
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, random_state=72, shuffle=True
)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(68, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation="sigmoid"))
# 이진분류에서는 마지막 아웃풋 레이어에 꼭 sigmoid 사용해야 됨

# 3. 컴파일, 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["mse", "accuracy"])
model.fit(x_train, y_train, epochs=500, batch_size=32)

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
# metrics에 mse, accuracy를 넣었기 때문에, evaluate 시 3가지 값이 나옴
print("loss:", loss)
print("mse:", mse)
print("accuracy:", accuracy)  # 1에 가까울수록 좋음

y_predict = model.predict(x_test)
# print(y_predict)
#y_predict2 = []
#for i in range(0, len(y_predict)):
#    y_predict2.append(1 if y_predict[i] > 0.5 else 0)

#y_predict2 = np.where(y_predict > 0.5, 1, 0)
y_predict2 = np.round(y_predict)

#print(y_predict2)
accuracy_score = accuracy_score(y_test, y_predict2)
print("accuracy_score:", accuracy_score)