import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 불러오기

header=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

data = pd.read_csv('./data/3.housing.csv',delim_whitespace=True,names=header)

# 상관 관계 행렬 시각화
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 각 특성의 분포 시각화
data.hist(figsize=(15, 12), bins=30, edgecolor='k')
plt.suptitle('Feature Distributions')
plt.show()