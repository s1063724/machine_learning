# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:25:13 2024

@author: 10305
"""
import pandas as pd
data = pd.read_csv('dataset/wine.csv', header=None)
df = data.copy()
# 第1~13欄(特徵值)
X = df.iloc[:, 1:]
# 欄位0紅酒分類(目標值)
y = df.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
#建立分類器
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
#預測結果
y_pred = clf.predict(X_test)
#print(X_test)
print(y_test)
print(y_pred)
#accuracy_score()分類準確評分
print(f"Accuracy： {accuracy_score(y_test, y_pred):.2f}")
cm = confusion_matrix(y_test, y_pred)
print(cm)

test1_pred = clf.predict([[1.51, 1.73, 1.98, 20.15, 85, 2.2, 1.92, .32, 1.48, 2.94, 1, 3.57, 172]])
print(test1_pred)

test2_pred = clf.predict([[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, .28, 2.29, 5.64, 1.04, 3.92, 1065]])
print(test2_pred)

test3_pred = clf.predict([[13.71, 5.65, 2.45, 20.5, 95, 1.68, .61, .52, 1.06, 7.7, .64, 1.74, 720]])
print(test3_pred)