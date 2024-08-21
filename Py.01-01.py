import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 불러오기
data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)

# 전처리: Inches -> cm, Pounds -> kg
data['Height(CM)'] = data['Height(Inches)'] * 2.54
data['Weight(KG)'] = data['Weight(Pounds)'] * 0.453592

# 데이터 배열로 변환
array = data[['Height(CM)', 'Weight(KG)']].values

X = array[:, 0]  # Height(CM)
Y = array[:, 1]  # Weight(KG)

X = X.reshape(-1, 1)  # X는 2D 배열이어야 함

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, Y_train)

# 예측
y_prediction = model.predict(X_test)

# 성능 평가
mae = mean_absolute_error(Y_test, y_prediction)
mse = mean_squared_error(Y_test, y_prediction)
rmse = np.sqrt(mse)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')

# 예측값과 실제값 시각화 (처음 100개 항목만)
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:100], Y_test[:100], color='blue', label='Actual values')
plt.scatter(X_test[:100], y_prediction[:100], color='red', label='Predicted values')
plt.plot(X_test[:100], y_prediction[:100], color='red', linewidth=2)
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Actual vs Predicted Weight (First 100 Samples)")
plt.legend()
plt.show()  # plt.show()는 함수 호출