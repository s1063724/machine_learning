# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 22:09:48 2024

@author: 10305
"""
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
# create dataframe from data in X_train 根據X_train中的資料創建dataframe
# label the columns using the strings in iris_dataset.feature_names 使用iris_dataset.feature_names中的字串標記列
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(f"Test set score: {model.score(X_test, y_test):.4f}")

inp = [[5, 2.9, 1, 0.2], [5.7, 2.8, 4.5, 1.2], [7.7, 3.8, 6.7, 2.1]]
inp_pred = model.predict(inp)
print(f"Predicted target name: {data.target_names[inp_pred]}")