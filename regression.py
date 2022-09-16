# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:42:28 2022

@author: Administrator
"""

import pandas as pd 
import numpy as np 
import statsmodels as st 
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

#df = pd.read_excel('/content/drive/MyDrive/수출량예측/Data/data_5.xlsx', header = 0)
df = pd.read_csv('val.csv')
df.info()
#%%
df.tail()
#%%
res = ols('total_export ~ date_time+feb+mar+exp_cn+exp_usa+exp_hk+imp_items+exp_heavy+exp_it+exp_world+kor_product_idx+mining_idx+construct_idx+service_idx+covid', data=df).fit()
res.summary()

#%%
res.params
sns.regplot(x='date', y='export1', data=df)
plt.show()
#%%
#pyplot과 seaborn으로 회귀그래프 
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=6, include_bias=True)

plt.rcParams["figure.figsize"] = (12, 6)
sns.regplot(x='date_time', y='exp_world', data=df)
plt.xlim(df['date_time'].min()-1, df['date_time'].max()+1)
plt.grid()
plt.show()
 
z=np.polyfit(df['date_time'], df['exp_world'], 1) # 기울기와 절편 확인
f=np.poly1d(z) # f(x): f함수에 x값을 넣으면 y값을 계산해 줌
print(z[0], z[1])
print(f(1))
 
#statsmodel을 통해 회귀식의 회귀계수(기울기, 절편) 확인
ols('exp_world ~ date_time+date_time^2+date_time^3+date_time^4+date_time^5+date_time^6', data=df).fit().summary()

#%%
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(264, 1), axis=0)
y = df['total_export']

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=5)
regr_2 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=10", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=3", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()