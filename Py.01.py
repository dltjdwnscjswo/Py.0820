import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 데이터 불러오기
data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)

# 데이터 확인
print(data.head())
print(data.columns)

# 전처리: Inches -> cm, Pounds -> kg
data['Height(CM)'] = data['Height(Inches)'] * 2.54
data['Weight(KG)'] = data['Weight(Pounds)'] * 0.453592

# 전처리 후 데이터 확인
print(data[['Height(CM)', 'Weight(KG)']].head())

# 데이터 요약 및 시각화
print(data.describe())

# 히스토그램 및 산점도
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data['Height(CM)'], bins=20, edgecolor='k')
plt.title('Height Distribution (CM)')
plt.xlabel('Height (CM)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(data['Weight(KG)'], bins=20, edgecolor='k')
plt.title('Weight Distribution (KG)')
plt.xlabel('Weight (KG)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 산점도: 키 vs 몸무게
plt.figure(figsize=(8, 6))
plt.scatter(data['Height(CM)'], data['Weight(KG)'], alpha=0.5)
plt.title('Height vs Weight')
plt.xlabel('Height (CM)')
plt.ylabel('Weight (KG)')
plt.show()

# 선형 회귀 모델 개발
X = data[['Height(CM)']]
y = data['Weight(KG)']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# 예측값과 실제값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual values')
plt.scatter(X_test, y_pred, color='red', label='Predicted values')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Actual vs Predicted Weight')
plt.xlabel('Height (CM)')
plt.ylabel('Weight (KG)')
plt.legend()
plt.show()