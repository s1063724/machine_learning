# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:41:27 2024

@author: 10305
"""
import pandas as pd


from sklearn.preprocessing import LabelEncoder 



data= pd.read_csv("dataset/NBApoints.csv")
df = data.copy()

le = LabelEncoder()
df.Pos =le.fit_transform(df.Pos)
df.Tm = le.fit_transform(df.Tm)

features = ['Pos', 'Age', 'Tm']
X = df.loc[:, features]
y = df.loc[:, '3P']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
#print(y_pred)
#inp = pd.DataFrame([5, 28, 10]).T
inp = [[5,28,10]]
#print(inp)
inp_pred = model.predict(inp)
print(f"三分球得球數= {inp_pred[0]:.4f}")

r_squared = model.score(X, y)
print(f"R_squared值= {r_squared:.4f}")

from sklearn.feature_selection import f_regression

pvalue = f_regression(X, y)[1]#0是f值，1是p值
print(f"f_regresstion\tP值= {pvalue}")
#判斷是否<0.05，小於則表顯著
#Pos
if(pvalue[0]<0.05):
    print("y")
#Age
if(pvalue[1]<0.05):
    print("y")
print('n')

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(mse,r2)