# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:09:52 2024

@author: 10305
"""
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])


# MEDV即預測目標向量
y = raw_df.values[1::2, 2] #有13個feature

#分出20%的資料作為test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Fit linear model 配適線性模型
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error
test_MAE = mean_absolute_error(y_test, model.predict(X_test))
test_MSE = mean_squared_error(y_test, model.predict(X_test))
test_RMSE = (test_MSE)**0.5
print(f'MAE: {test_MAE:.4f}')
print(f'MSE: {test_MSE:.4f}')
print(f'RMSE: {test_RMSE:.4f}')

inp = [[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]]
print(f"預測房價: {model.predict(inp)[0]:.4f}")