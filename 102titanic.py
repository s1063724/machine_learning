# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:21:32 2024

@author: 10305
"""
import pandas as pd
from sklearn import preprocessing, linear_model


# 原始資料
titanic = pd.read_csv("dataset/titanic.csv")
df = titanic.copy()

# 將年齡的空值填入年齡的中位數

#df.info()
age_median = df.Age.median()
#fillna()填補空值
#   inplace=True 直接取代
df.Age.fillna(age_median, inplace=True)
#df.info()
print(f"年齡中位數= {age_median:.0f}")
# 轉換欄位值成為數值
le = preprocessing.LabelEncoder()
df.Pclass = le.fit_transform(df.Pclass)
df.Sex = le.fit_transform(df.Sex)
X = df.loc[:, ['Pclass', 'Age', 'Sex']]
y = df.loc[:, 'Survived']

# 建立模型(羅吉斯回歸)
model = linear_model.LogisticRegression()
model.fit(X, y)
intercept = model.intercept_
coef = model.coef_
print(coef)#選第3個sex作為迴歸係數
print(intercept)
print(f'迴歸係數= {coef[0][2]:.4f}')
print(f'截距= {intercept[0]:.4f}')


# 混淆矩陣(Confusion Matrix)，計算準確度
accuracy = model.score(X, y)
print(f'accuracy: {accuracy:.4f}')